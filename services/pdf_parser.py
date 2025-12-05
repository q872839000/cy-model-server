from __future__ import annotations

import os
import html
from io import BytesIO
from typing import Optional, List, Tuple
from dataclasses import dataclass
from statistics import median
import re
from collections import defaultdict
import numpy as np

from loguru import logger
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from services.document_parser import BaseDocumentParser
from models.parsed_document import (
    ParsedDocument,
    SectionNode,
    Block,
    BlockType,
    Coordinates,
    DocumentMetadata,
    TableData,
    Styles,
    Chunk,
    ChunkType,
    ChunkMeta
)
from core.config import load_settings
import yaml

# 可选依赖：中文分词（如果未安装则使用简单分词）
try:
    import jieba
except ImportError:
    jieba = None


@dataclass
class _LinearBlock:
    """内部使用的线性块数据结构，用于解析过程中的中间表示。

    属性:
        block: 最终的 Block 对象
        page: 所在页码（从 1 开始）
        bbox: 边界框 (x0, y0, x1, y1)，用于几何计算
        font_size: 字体大小（用于标题识别和段落合并）
        is_text: 是否为文本块（True=段落，False=图片/表格）
        global_order: 全局顺序编号
        col_id: 列 ID（多列布局时使用，None 表示未分配）
    """
    block: Block
    page: int
    bbox: Optional[tuple[float, float, float, float]]
    font_size: float
    is_text: bool
    global_order: int
    col_id: Optional[int] = None


class PdfDocumentParser(BaseDocumentParser):
    """PDF 文档解析器：将 PDF 二进制内容解析为 ParsedDocument 结构。

    企业级实现：
    - 使用 XGBoost 模型进行段落拼接决策
    - 使用 KMeans 进行列聚类
    - 支持 OCR、表格识别、多列布局
    - 基于字体大小分布与标题编号模式，自动识别 H1/H2/H3 级标题
    - 按照线性顺序构建多级 SectionNode 树，非标题块挂载到最近的上级章节下
    """

    def __init__(self, model_path: Optional[str] = None):
        """初始化解析器，从配置文件加载 XGBoost 模型。

        Args:
            model_path: updown_concat_xgb.model 的路径（可选，会覆盖配置文件中的设置）。
                        如果为 None，则从 config.yaml 的 pdf_parser.updown_concat_model_path 读取。
        """
        # 从配置文件读取 PDF 解析器配置
        settings = load_settings()
        cfg = yaml.safe_load(open(settings.models_config_path, 'r', encoding='utf-8')) or {}
        pdf_parser_cfg = cfg.get('pdf_parser') or {}

        # 优先使用传入的 model_path，否则从配置文件读取
        if model_path is None:
            model_path = pdf_parser_cfg.get('updown_concat_model_path', '')

        # OCR 语言设置（从配置文件读取，默认英文）
        self.ocr_lang = pdf_parser_cfg.get('ocr_lang', 'eng')

        # 加载 XGBoost 段落拼接模型
        self.updown_concat_model: Optional[object] = None
        if model_path and os.path.exists(model_path):
            try:
                # 尝试加载模型，根据文件扩展名选择适当的方法
                file_ext = os.path.splitext(model_path)[1].lower()
                
                # 创建 XGBoost 模型对象
                self.updown_concat_model = xgb.Booster()
                
                # 尝试直接加载（适用于所有格式，但可能在旧格式上失败）
                try:
                    self.updown_concat_model.load_model(model_path)
                    model_format = "未知"  # 无法直接确定，默认描述
                    
                    # 根据扩展名推断格式
                    if file_ext == '.json':
                        model_format = "JSON"
                    elif file_ext == '.ubj':
                        model_format = "UBJ"
                    elif file_ext == '.model':
                        # 这可能是旧的二进制格式，但如果加载成功，则可能是其他格式
                        model_format = "可能是旧二进制格式或其他格式"
                    
                    logger.info("已加载段落拼接模型: {} (格式: {})", model_path, model_format)
                except Exception as e:
                    # 如果直接加载失败，尝试其他方法
                    logger.debug("直接加载模型失败，尝试其他方法: {}", e)
                    
                    # 检查错误是否与旧二进制格式相关
                    if "binary format" in str(e).lower() or "deprecated" in str(e).lower():
                        logger.warning("检测到旧的二进制模型格式，建议迁移到UBJ或JSON格式")
                        # 尝试使用XGBoost 3.0+的加载方式（通过API可能不同）
                        self.updown_concat_model = xgb.Booster()
                        self.updown_concat_model.load_model(model_path)
                        logger.info("成功加载模型: {} (使用兼容性模式)", model_path)
                    else:
                        # 其他错误，重新抛出
                        raise
            except Exception as e:
                logger.warning("加载段落拼接模型失败: {}，将使用规则判断", e)
                # 确保模型对象为None，表示加载失败
                self.updown_concat_model = None
        else:
            logger.warning("未找到段落拼接模型（路径: {}），将使用规则判断", model_path or "未配置")

    def supports(self, filename: str, content_type: Optional[str] = None) -> bool:
        """检查是否支持该文件类型。

        Args:
            filename: 文件名
            content_type: MIME 类型（可选）

        Returns:
            如果支持 PDF 格式则返回 True
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return True
        if content_type:
            ct = content_type.lower()
            if "pdf" in ct:
                return True
        return False

    def parse(
            self,
            content: bytes,
            filename: str,
            content_type: Optional[str] = None,
    ) -> ParsedDocument:
        """解析 PDF 文件，生成结构化的 ParsedDocument。

        解析流程：
        1. 使用 PyMuPDF 提取文本和图片块
        2. 使用 pdfplumber 提取表格
        3. 可选：对图片进行 OCR
        4. 过滤页眉页脚和目录页
        5. 进行列聚类（多列布局）
        6. 段落合并（使用 XGBoost 模型或规则）
        7. 构建章节树（SectionNode）
        8. 生成最终文档结构

        Args:
            content: PDF 文件的二进制内容
            filename: 文件名
            content_type: MIME 类型（可选）

        Returns:
            ParsedDocument: 解析后的文档结构

        Raises:
            RuntimeError: 如果 PyMuPDF 未安装或 PDF 打开失败
        """
        pdf = self._open_pdf(content)
        try:
            # 记录每页的尺寸（用于列聚类）
            page_sizes: dict[int, Tuple[float, float]] = {}

            page_count = pdf.page_count
            logger.info("开始解析 PDF 文件: name={}, pages={}", filename, page_count)

            # 收集页面尺寸信息
            for page_index in range(page_count):
                page = pdf.load_page(page_index)
                page_number = page_index + 1
                page_sizes[page_number] = (float(page.rect.width), float(page.rect.height))

            # 步骤 1: 使用 PyMuPDF 提取文本和图片块
            linear_blocks = self._extract_blocks_with_pymupdf(pdf)

            # 步骤 2: 使用 pdfplumber 提取表格
            next_order = max((lb.global_order for lb in linear_blocks), default=0) + 1
            table_blocks = self._extract_tables_with_pdfplumber(content, next_order)
            if table_blocks:
                linear_blocks.extend(table_blocks)
                # 移除表格区域内的文本块（避免重复）
                linear_blocks = self._remove_text_blocks_in_tables(linear_blocks)

            # 步骤 3: 可选 OCR（如果安装了 pytesseract）
            self._attach_image_ocr(pdf, linear_blocks)

            # 步骤 4: 检测并过滤页眉页脚
            header_texts, footer_texts, page_max_y = self._detect_header_footer_candidates(
                linear_blocks,
                page_count,
            )
            linear_blocks = self._filter_headers_footers(
                linear_blocks,
                header_texts,
                footer_texts,
                page_max_y,
            )

            # 步骤 5: 检测并过滤目录页
            toc_pages = self._detect_toc_pages(linear_blocks, page_count)
            if toc_pages:
                linear_blocks = [lb for lb in linear_blocks if lb.page not in toc_pages]

            # 步骤 6: 列聚类（多列布局恢复阅读顺序）
            linear_blocks = self._assign_columns(linear_blocks, page_sizes)
            linear_blocks = self._sort_linear_blocks(linear_blocks)

            # 步骤 7: 段落合并（使用 XGBoost 模型或规则）
            body_font_size_for_merge = self._estimate_body_font_size(linear_blocks)
            if body_font_size_for_merge is not None:
                linear_blocks = self._merge_paragraphs(
                    linear_blocks,
                    body_font_size_for_merge,
                )
                linear_blocks = self._sort_linear_blocks(linear_blocks)

            # 步骤 8: 构建章节树
            sections = self._build_section_tree(linear_blocks)

            # 步骤 9: 生成全文和元数据
            full_text = self._build_full_text(linear_blocks)
            metadata_title = os.path.basename(filename) or None
            # 尝试从第一级标题提取文档标题
            if sections:
                first_level_title = next(
                    (sec.title for sec in sections if sec.level == 1 and sec.title.strip()),
                    None,
                )
                if first_level_title:
                    metadata_title = first_level_title
            metadata = DocumentMetadata(title=metadata_title)

            # 步骤 10: 进行语义分块，生成与SectionNode平级的Chunk结构
            chunks = self._semantic_chunking(sections, filename, content_type or "application/pdf")

            parsed = ParsedDocument(
                original_filename=filename,
                content_type=content_type or "application/pdf",
                metadata=metadata,
                text=full_text,
                structure_tree=sections,
                chunks=chunks,
            )

            logger.info(
                "PDF 解析完成: name={}, pages={}, sections={}, blocks={}, chunks={}, full_text_length={}",
                filename,
                page_count,
                len(sections),
                sum(len(s.content_blocks) for s in sections),
                len(chunks),
                len(full_text)
            )
            return parsed
        finally:
            try:
                pdf.close()
            except Exception:
                pass

    def _extract_blocks_with_pymupdf(self, pdf) -> List[_LinearBlock]:
        """使用 PyMuPDF 提取文本和图片块。

        从 PDF 中提取所有文本块（type=0）和图片块（type=1），
        保留坐标、字体、样式等信息。

        Args:
            pdf: PyMuPDF 的 Document 对象

        Returns:
            提取的 _LinearBlock 列表
        """
        linear_blocks: List[_LinearBlock] = []
        global_order = 1
        for page_index in range(pdf.page_count):
            page = pdf.load_page(page_index)
            page_number = page_index + 1
            blocks_info = page.get_text("dict").get("blocks", [])
            for block in blocks_info:
                block_type = block.get("type", 0)
                bbox = block.get("bbox")
                coords: Optional[Coordinates] = None
                if bbox is not None and len(bbox) >= 4:
                    x0, y0, x1, y1 = bbox[:4]
                    coords = Coordinates(
                        page=page_number,
                        x=float(x0),
                        y=float(y0),
                        w=float(x1 - x0),
                        h=float(y1 - y0),
                    )
                if block_type == 0:
                    text_fragments: List[str] = []
                    font_sizes: List[float] = []
                    flags_list: List[int] = []
                    fonts: List[str] = []
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            span_text = span.get("text")
                            if span_text:
                                text_fragments.append(span_text)
                            size = span.get("size")
                            if isinstance(size, (int, float)):
                                font_sizes.append(float(size))
                            flags = span.get("flags")
                            if isinstance(flags, int):
                                flags_list.append(flags)
                            font = span.get("font")
                            if isinstance(font, str):
                                fonts.append(font)
                    text = "".join(text_fragments).strip()
                    if not text:
                        continue
                    max_font_size = max(font_sizes) if font_sizes else 0.0
                    styles = Styles(
                        font_size=max_font_size or None,
                        bold=self._span_flags_any(flags_list, 2),
                        italic=self._span_flags_any(flags_list, 4),
                        font_family=fonts[0] if fonts else None,
                        indent=float(bbox[0]) if bbox else None,
                    )
                    block_id = f"blk-p{page_number:04d}-{global_order:04d}"
                    block_obj = Block(
                        block_id=block_id,
                        type=BlockType.PARAGRAPH,
                        order=global_order,
                        text=text,
                        html=self._wrap_paragraph_html(text),
                        ocr_text=None,
                        table=None,
                        coordinates=coords,
                        styles=styles,
                        metadata={},
                    )
                    linear_blocks.append(
                        _LinearBlock(
                            block=block_obj,
                            page=page_number,
                            bbox=tuple(bbox) if bbox is not None else None,
                            font_size=max_font_size,
                            is_text=True,
                            global_order=global_order,
                        ),
                    )
                    global_order += 1
                elif block_type == 1:
                    block_id = f"img-p{page_number:04d}-{global_order:04d}"
                    block_obj = Block(
                        block_id=block_id,
                        type=BlockType.IMAGE,
                        order=global_order,
                        text=None,
                        html=None,
                        ocr_text=None,
                        table=None,
                        coordinates=coords,
                        styles=None,
                        metadata={"source": "image"},
                    )
                    linear_blocks.append(
                        _LinearBlock(
                            block=block_obj,
                            page=page_number,
                            bbox=tuple(bbox) if bbox is not None else None,
                            font_size=0.0,
                            is_text=False,
                            global_order=global_order,
                        ),
                    )
                    global_order += 1
        return linear_blocks

    def _span_flags_any(self, flags: List[int], mask: int) -> bool:
        return any((flag & mask) == mask for flag in flags)

    def _wrap_paragraph_html(self, text: str) -> str:
        if not text:
            return ""
        return f"<p>{html.escape(text)}</p>"

    def _extract_tables_with_pdfplumber(self, content: bytes, start_order: int) -> List[_LinearBlock]:
        """使用 pdfplumber 提取表格。

        从 PDF 中检测并提取所有表格，转换为 TableData 结构。

        Args:
            content: PDF 文件的二进制内容
            start_order: 起始顺序编号

        Returns:
            提取的表格块列表
        """
        table_blocks: List[_LinearBlock] = []
        current_order = start_order
        try:
            with pdfplumber.open(BytesIO(content)) as plumber_pdf:
                for page_index, page in enumerate(plumber_pdf.pages):
                    tables = page.find_tables(table_settings=self._pdfplumber_table_settings())
                    if not tables:
                        continue
                    for table in tables:
                        rows = table.extract()
                        if not rows:
                            continue
                        x0, top, x1, bottom = table.bbox
                        coords = Coordinates(
                            page=page_index + 1,
                            x=float(x0),
                            y=float(top),
                            w=float(x1 - x0),
                            h=float(bottom - top),
                        )
                        rows_clean = [[cell or "" for cell in row] for row in rows]
                        html_table = self._table_rows_to_html(rows_clean)
                        block = Block(
                            block_id=f"tbl-p{page_index + 1:04d}-{current_order:04d}",
                            type=BlockType.TABLE,
                            order=current_order,
                            text=None,
                            html=html_table,
                            ocr_text=None,
                            table=TableData(rows=rows_clean),
                            coordinates=coords,
                            styles=None,
                            metadata={"source": "table"},
                        )
                        table_blocks.append(
                            _LinearBlock(
                                block=block,
                                page=page_index + 1,
                                bbox=(float(x0), float(top), float(x1), float(bottom)),
                                font_size=0.0,
                                is_text=False,
                                global_order=current_order,
                            )
                        )
                        current_order += 1
        except Exception as exc:
            logger.warning("表格提取失败: {}", exc)
        return table_blocks

    def _remove_text_blocks_in_tables(self, blocks: List[_LinearBlock]) -> List[_LinearBlock]:
        """移除表格区域内的文本块，避免重复内容。

        当表格被提取后，表格区域内的文本块应该被移除，
        因为表格已经包含了这些文本内容。

        Args:
            blocks: 所有内容块列表

        Returns:
            过滤后的块列表（移除了表格区域内的文本块）
        """
        table_regions = [b for b in blocks if b.block.type == BlockType.TABLE and b.bbox]
        if not table_regions:
            return blocks
        filtered: List[_LinearBlock] = []
        for lb in blocks:
            if lb.block.type == BlockType.PARAGRAPH and lb.bbox:
                # 如果文本块与表格区域重叠超过 40%，则移除
                overlapped = any(
                    self._bbox_overlap(lb.bbox, tbl.bbox) > 0.4 for tbl in table_regions  # type: ignore[arg-type]
                )
                if overlapped:
                    continue
            filtered.append(lb)
        return filtered

    def _bbox_overlap(
            self,
            a: tuple[float, float, float, float],
            b: tuple[float, float, float, float],
    ) -> float:
        """计算两个边界框的重叠率（IoU 的简化版本）。

        目的：判断文本块是否在表格区域内，用于去重。
        当表格被提取后，需要移除表格区域内的文本块，避免内容重复。
        重叠率 = 交集面积 / 较小框的面积，值越大表示重叠越多。

        Args:
            a: 第一个边界框 (x0, y0, x1, y1)，通常是文本块
            b: 第二个边界框 (x0, y0, x1, y1)，通常是表格块

        Returns:
            重叠率（0.0 表示不重叠，1.0 表示完全重叠）
        """
        # 解包边界框坐标：左上角 (x0, y0) 和右下角 (x1, y1)
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b

        # 计算两个框的交集区域
        # 交集的左上角：取两个框左上角坐标的最大值（更靠右、更靠下）
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        # 交集的右下角：取两个框右下角坐标的最小值（更靠左、更靠上）
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)

        # 如果交集无效（右下角在左上角左边或上边），说明两个框不重叠
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0

        # 计算交集面积
        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        # 计算两个框各自的面积（使用 1e-6 避免除零）
        area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
        area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)

        # 返回重叠面积与较小框面积的比值
        # 这样即使文本块很大，只要与表格重叠足够多，也会被识别为重叠
        return inter_area / min(area_a, area_b)

    def _table_rows_to_html(self, rows: List[List[str]]) -> str:
        if not rows:
            return ""
        parts = ["<table>"]
        for row in rows:
            parts.append("<tr>")
            for cell in row:
                parts.append(f"<td>{html.escape(str(cell))}</td>")
            parts.append("</tr>")
        parts.append("</table>")
        return "".join(parts)

    def _attach_image_ocr(self, pdf, linear_blocks: List[_LinearBlock]) -> None:
        """为图片块附加 OCR 文本。

        对图片块进行 OCR 识别，将识别结果写入 block.ocr_text。

        Args:
            pdf: PyMuPDF 的 Document 对象
            linear_blocks: 内容块列表（会原地修改）
        """
        # 使用配置文件中设置的 OCR 语言
        lang = self.ocr_lang
        for lb in linear_blocks:
            if lb.block.type != BlockType.IMAGE:
                continue
            if lb.block.ocr_text:  # 已有 OCR 文本则跳过
                continue
            if not lb.bbox:
                continue
            try:
                # 从 PDF 页面裁剪图片区域
                page = pdf.load_page(lb.page - 1)
                rect = fitz.Rect(*lb.bbox)
                # 使用 2x 缩放提高 OCR 准确率
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect, alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                # 执行 OCR
                text = pytesseract.image_to_string(image, lang=lang).strip()
                lb.block.ocr_text = text or None
            except pytesseract.TesseractNotFoundError:
                logger.debug("图片 OCR 失败: tesseract 未安装或不在 PATH 中")
            except Exception as exc:  # pragma: no cover - OCR 失败不阻塞
                logger.debug("图片 OCR 失败: {}", exc)

    def _pdfplumber_table_settings(self) -> dict:
        return {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_y_tolerance": 5,
            "intersection_x_tolerance": 5,
            "min_words_vertical": 2,
            "min_words_horizontal": 2,
        }

    def _build_section_tree(self, linear_blocks: List[_LinearBlock]) -> List[SectionNode]:
        """根据线性 block 列表构建多级章节树。

        目的：将线性的内容块组织成层次化的章节结构（SectionNode 树），
        用于前端目录展示、结构感知分块、原文精确定位等。

        策略：
        1. 估计正文字体大小（中位数），大于正文显著的字体视为候选标题字体
        2. 结合字体大小和编号模式（"1 / 1.1 / 1.1.1"）识别 H1/H2/H3 级标题
        3. 使用栈维护当前章节层级，将非标题块挂到最近的上级章节

        Args:
            linear_blocks: 已排序的内容块列表

        Returns:
            章节树根节点列表（可能有多个根节点，但通常只有一个）
        """
        # 按阅读顺序排序（页 -> 列 -> Y坐标 -> 顺序）
        ordered_blocks = self._sort_linear_blocks(linear_blocks)
        # 筛选出文本块（用于字体大小统计和标题识别）
        text_blocks = [
            b
            for b in ordered_blocks
            if b.is_text and b.font_size > 0 and (b.block.text or "").strip()
        ]

        # 如果没有可用文本块，创建一个根节点包裹所有内容
        if not text_blocks:
            if not ordered_blocks:
                return []
            first_page = min(b.page for b in ordered_blocks)
            last_page = max(b.page for b in ordered_blocks)
            root = SectionNode(
                section_id="sec-root",
                parent_id=None,
                level=1,
                title="Document",
                order=1,
                page_start=first_page,
                page_end=last_page,
                content_blocks=[b.block for b in ordered_blocks],
                children=[],
            )
            return [root]

        # ========== 估计正文字体大小 ==========
        # 使用中位数作为正文字体大小（比平均值更稳健，不受极端值影响）
        font_sizes = [b.font_size for b in text_blocks]
        try:
            body_font_size = float(median(font_sizes))
        except Exception:
            body_font_size = float(font_sizes[0])  # 如果计算失败，使用第一个值

        # ========== 建立字体大小到标题层级的映射 ==========
        # 从大到小排序不同字体大小，选出若干个作为候选标题等级
        # 例如：字体 18pt -> H1，字体 14pt -> H2，字体 12pt -> H3
        unique_sizes = sorted(set(font_sizes), reverse=True)
        levels_by_size: dict[float, int] = {}  # {字体大小: 标题层级}
        level = 1
        for size in unique_sizes:
            if level > 3:  # 最多支持 3 级标题
                break
            # 如果字体大小比正文大 5% 或绝对值大 1pt，视为标题字体
            if size >= body_font_size * 1.05 or size - body_font_size >= 1.0:
                levels_by_size[size] = level
                level += 1

        # ========== 构建章节树 ==========
        sections: List[SectionNode] = []  # 根节点列表
        stack: List[SectionNode] = []  # 章节栈（维护当前路径，栈顶是最内层章节）
        section_order = 1  # 章节顺序编号

        # 遍历所有块，识别标题并构建树结构
        for lb in ordered_blocks:
            heading_level: Optional[int] = None
            # 只对文本块进行标题识别
            if lb.is_text and lb.block.text:
                heading_level = self._detect_heading_level(
                    lb,
                    body_font_size,
                    levels_by_size,
                )

            if heading_level:
                # ========== 发现标题，创建新的章节节点 ==========
                section = SectionNode(
                    section_id=f"sec-{lb.block.block_id}",
                    parent_id=None,  # 稍后会设置
                    level=heading_level,
                    title=lb.block.text or "",
                    order=section_order,
                    page_start=lb.page,
                    page_end=lb.page,
                    content_blocks=[],  # 标题本身不包含在 content_blocks 中
                    children=[],
                )
                section_order += 1

                # 弹出栈中比当前层级更深或相同的章节
                # 例如：当前是 H2，栈中有 H3，需要弹出 H3，让 H2 成为 H3 的兄弟
                while stack and stack[-1].level >= heading_level:
                    stack.pop()

                # 设置父子关系
                if not stack:
                    # 栈为空，说明这是根级别的章节
                    sections.append(section)
                else:
                    # 栈不为空，当前章节是栈顶章节的子节点
                    parent = stack[-1]
                    section.parent_id = parent.section_id
                    parent.children.append(section)

                # 将当前章节压入栈（成为新的最内层章节）
                stack.append(section)
            else:
                # ========== 普通内容块，挂到当前最内层章节 ==========
                # 如果还没有章节，创建一个虚拟根节点
                if not stack:
                    root = SectionNode(
                        section_id="sec-root",
                        parent_id=None,
                        level=1,
                        title="Document",
                        order=1,
                        page_start=lb.page,
                        page_end=lb.page,
                        content_blocks=[],
                        children=[],
                    )
                    sections.append(root)
                    stack.append(root)

                # 将内容块添加到当前最内层章节
                current = stack[-1]
                current.content_blocks.append(lb.block)
                # 更新章节的页码范围
                if current.page_start is None or lb.page < current.page_start:
                    current.page_start = lb.page
                if current.page_end is None or lb.page > current.page_end:
                    current.page_end = lb.page

        return sections

    def _estimate_body_font_size(self, linear_blocks: List[_LinearBlock]) -> Optional[float]:
        """估计正文字体大小。

        使用所有文本块字体大小的中位数作为正文字体大小，
        用于区分标题和正文。

        Args:
            linear_blocks: 内容块列表

        Returns:
            正文字体大小（如果无法估计则返回 None）
        """
        text_blocks = [
            b
            for b in linear_blocks
            if b.is_text and b.font_size > 0 and (b.block.text or "").strip()
        ]
        if not text_blocks:
            return None
        font_sizes = [b.font_size for b in text_blocks]
        try:
            return float(median(font_sizes))
        except Exception:
            return float(font_sizes[0])

    def _assign_columns(
            self,
            linear_blocks: List[_LinearBlock],
            page_sizes: dict[int, Tuple[float, float]],
    ) -> List[_LinearBlock]:
        """基于 KMeans 聚类进行列检测，恢复阅读顺序。

        目的：识别多列布局的 PDF，为每个文本块分配列 ID（col_id），
        确保后续排序时能按正确的阅读顺序（先左列后右列，列内从上到下）处理。

        方法：使用 KMeans 对文本块的 x0 坐标进行聚类，每个聚类代表一列。
        通过 silhouette_score 选择最优的列数（k 值）。

        Args:
            linear_blocks: 待处理的块列表
            page_sizes: 每页的尺寸 (width, height)

        Returns:
            已分配 col_id 的块列表（原地修改）
        """
        # 按页分组，只处理文本块（图片和表格不需要列检测）
        per_page: dict[int, List[_LinearBlock]] = defaultdict(list)
        for block in linear_blocks:
            if block.is_text and block.bbox is not None:
                per_page[block.page].append(block)

        # 逐页进行列检测
        for page, blocks in per_page.items():
            if not blocks:
                continue

            # 获取页面宽度（用于计算缩进容差）
            page_width = page_sizes.get(page, (0.0, 0.0))[0] or 0.0
            if page_width <= 0:
                continue

            # 提取所有块的左边界 x0 坐标，作为聚类的特征
            x0s_raw = np.array([b.bbox[0] for b in blocks if b.bbox], dtype=float)
            if len(x0s_raw) == 0:
                continue

            # 计算页面内容区域的最小和最大 x 坐标
            min_x0 = np.min(x0s_raw)  # 最左侧块的左边界
            max_x1 = np.max([b.bbox[2] for b in blocks if b.bbox])  # 最右侧块的右边界
            width = max_x1 - min_x0  # 内容区域宽度

            # 处理缩进：将接近页面左边缘的块统一归到 min_x0
            # 这样可以避免因为轻微缩进导致被误分为不同列
            # 容差设为内容宽度的 12%（参考 RAGFlow）
            INDENT_TOL = width * 0.12
            x0s = []
            for x in x0s_raw:
                # 如果块的左边界接近页面左边缘（在容差范围内），统一归到 min_x0
                if abs(x - min_x0) < INDENT_TOL:
                    x0s.append([min_x0])
                else:
                    # 否则使用原始 x0 坐标
                    x0s.append([x])
            x0s = np.array(x0s, dtype=float)

            # 使用 KMeans 聚类进行列检测（需要至少 2 个块）
            if len(blocks) >= 2:
                # 尝试不同的 k 值（列数），最多尝试 4 列或块数量（取较小值）
                max_try = min(4, len(blocks))
                if max_try < 2:
                    max_try = 1
                best_k = 1  # 最佳列数
                best_score = -1  # 最佳轮廓系数

                # 遍历不同的 k 值，选择轮廓系数最高的
                for k in range(1, max_try + 1):
                    try:
                        # 使用 KMeans 进行聚类
                        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                        labels = km.fit_predict(x0s)  # 获取每个块所属的聚类标签
                        centers = np.sort(km.cluster_centers_.flatten())  # 聚类中心（按 x 坐标排序）

                        # 计算轮廓系数（silhouette score）评估聚类质量
                        # 轮廓系数越高，说明聚类效果越好（块之间的分离度越高）
                        if len(centers) > 1:
                            try:
                                score = silhouette_score(x0s, labels)
                            except ValueError:
                                continue
                        else:
                            # 只有 1 个聚类时，轮廓系数为 0
                            score = 0

                        # 记录最佳 k 值和对应的轮廓系数
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except Exception:
                        continue

                # 使用最佳 k 值进行最终聚类并分配列 ID
                if best_k > 0:
                    try:
                        # 使用最佳 k 值重新聚类
                        km = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
                        labels = km.fit_predict(x0s)
                        centers = km.cluster_centers_.flatten()
                        # 按聚类中心的 x 坐标排序，确定列的左右顺序
                        order = np.argsort(centers)  # 例如：[1, 0, 2] 表示中心 1 最左，中心 0 中间，中心 2 最右
                        # 重新映射：将聚类标签映射到列 ID（0=最左列，1=中间列，2=最右列...）
                        remap = {orig: new for new, orig in enumerate(order)}

                        # 为每个块分配列 ID
                        for i, b in enumerate(blocks):
                            if b.bbox:
                                # 根据聚类标签获取列 ID
                                b.col_id = remap.get(labels[i], 0)
                        logger.debug("[Page {}] 列聚类: k={}, score={:.2f}", page, best_k, best_score)
                        continue
                    except Exception as e:
                        logger.debug("KMeans 聚类失败，回退到简单方法: {}", e)

            # 回退到简单聚类方法（当 KMeans 失败或块数量太少时使用）
            # 使用基于距离的简单聚类：将 x 坐标相近的块归为一列
            indent_tol = max(25.0, page_width * 0.08)  # 列间距离容差（至少 25px 或页面宽度的 8%）
            clusters: List[tuple[float, List[_LinearBlock]]] = []  # (列中心 x 坐标, 该列的块列表)

            # 按 x 坐标从左到右排序，逐个处理
            for b in sorted(blocks, key=lambda blk: (blk.bbox[0] if blk.bbox else 0.0)):
                if not b.bbox:
                    continue
                # 计算块的中心 x 坐标
                center = (b.bbox[0] + b.bbox[2]) / 2
                placed = False

                # 尝试将当前块加入到已有的列中
                for idx, (mean, members) in enumerate(clusters):
                    # 如果块的中心 x 坐标与列中心距离在容差内，归入该列
                    if abs(center - mean) <= indent_tol:
                        members.append(b)
                        # 更新列中心（使用平均值）
                        new_mean = (mean * (len(members) - 1) + center) / len(members)
                        clusters[idx] = (new_mean, members)
                        placed = True
                        break

                # 如果无法归入已有列，创建新列
                if not placed:
                    clusters.append((center, [b]))

            # 按列中心 x 坐标排序（从左到右）
            clusters.sort(key=lambda item: item[0])
            # 为每个块分配列 ID（0=最左列，1=中间列，2=最右列...）
            for col_id, (_, members) in enumerate(clusters):
                for member in members:
                    member.col_id = col_id

        return linear_blocks

    def _sort_linear_blocks(self, linear_blocks: List[_LinearBlock]) -> List[_LinearBlock]:
        """对块列表进行排序，恢复正确的阅读顺序。

        排序规则（优先级从高到低）：
        1. 页码（page）
        2. 列 ID（col_id，多列布局时）
        3. Y 坐标（从上到下）
        4. 全局顺序（global_order，作为稳定排序的辅助）

        Args:
            linear_blocks: 待排序的块列表

        Returns:
            排序后的块列表
        """
        return sorted(
            linear_blocks,
            key=lambda b: (
                b.page,
                b.col_id if b.col_id is not None else 0,
                b.bbox[1] if b.bbox is not None else 0.0,
                b.global_order,
            ),
        )

    def _normalize_header_footer_text(self, text: str) -> str:
        return " ".join(text.split())

    def _detect_header_footer_candidates(
            self,
            linear_blocks: List[_LinearBlock],
            page_count: int,
    ) -> tuple[set[str], set[str], dict[int, float]]:
        if page_count <= 0 or not linear_blocks:
            return set(), set(), {}

        per_page: dict[int, List[_LinearBlock]] = {}
        page_max_y: dict[int, float] = {}

        for lb in linear_blocks:
            if lb.bbox is not None:
                _, _, _, y1 = lb.bbox
                prev_max = page_max_y.get(lb.page)
                if prev_max is None or y1 > prev_max:
                    page_max_y[lb.page] = y1
            if not lb.is_text or not lb.block.text or lb.bbox is None:
                continue
            per_page.setdefault(lb.page, []).append(lb)

        header_counts: dict[str, int] = defaultdict(int)
        footer_counts: dict[str, int] = defaultdict(int)

        for page, blocks in per_page.items():
            if not blocks:
                continue
            sorted_by_top = sorted(
                blocks,
                key=lambda b: b.bbox[1] if b.bbox is not None else 0.0,
            )
            top_candidates = sorted_by_top[:3]
            sorted_by_bottom = sorted(
                blocks,
                key=lambda b: b.bbox[3] if b.bbox is not None else 0.0,
            )
            bottom_candidates = sorted_by_bottom[-3:]

            for lb in top_candidates:
                text = self._normalize_header_footer_text(lb.block.text or "")
                if len(text) >= 3:
                    header_counts[text] += 1
            for lb in bottom_candidates:
                text = self._normalize_header_footer_text(lb.block.text or "")
                if len(text) >= 3:
                    footer_counts[text] += 1

        min_repeat = max(2, int(page_count * 0.5)) if page_count > 0 else 2
        header_texts = {t for t, c in header_counts.items() if c >= min_repeat}
        footer_texts = {t for t, c in footer_counts.items() if c >= min_repeat}

        return header_texts, footer_texts, page_max_y

    def _is_footer_page_number(
            self,
            normalized_text: str,
            page: int,
            bbox: tuple[float, float, float, float],
            page_max_y: dict[int, float],
    ) -> bool:
        if not normalized_text:
            return False
        if not re.fullmatch(r"\d{1,4}", normalized_text):
            return False
        max_y = page_max_y.get(page)
        if max_y is None or max_y <= 0:
            return False
        y_bottom = bbox[3]
        threshold = max(20.0, max_y * 0.05)
        return max_y - y_bottom <= threshold

    def _filter_headers_footers(
            self,
            linear_blocks: List[_LinearBlock],
            header_texts: set[str],
            footer_texts: set[str],
            page_max_y: dict[int, float],
    ) -> List[_LinearBlock]:
        if not linear_blocks:
            return []
        filtered: List[_LinearBlock] = []
        for lb in linear_blocks:
            if lb.is_text and lb.block.text:
                norm = self._normalize_header_footer_text(lb.block.text)
                if norm in header_texts or norm in footer_texts:
                    continue
                if (
                        lb.bbox is not None
                        and self._is_footer_page_number(norm, lb.page, lb.bbox, page_max_y)
                ):
                    continue
            filtered.append(lb)
        return filtered

    def _is_toc_item_text(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 4:
            return False
        if "..." in stripped or "···" in stripped:
            return True
        m = re.match(r"^\d+(?:\.\d+)*[\. 、\s].+\s+\d{1,4}$", stripped)
        if m:
            return True
        return False

    def _detect_toc_pages(
            self,
            linear_blocks: List[_LinearBlock],
            page_count: int,
    ) -> set[int]:
        if page_count <= 0 or not linear_blocks:
            return set()
        max_toc_page = min(page_count, 5)
        per_page_counts: dict[int, int] = defaultdict(int)
        for lb in linear_blocks:
            if not lb.is_text or not lb.block.text:
                continue
            if lb.page > max_toc_page:
                continue
            if self._is_toc_item_text(lb.block.text):
                per_page_counts[lb.page] += 1
        toc_pages: set[int] = set()
        for page, count in per_page_counts.items():
            if count >= 5:
                toc_pages.add(page)
        return toc_pages

    def _simple_tokenize(self, text: str) -> List[str]:
        """简单的分词函数，用于特征提取。

        优先使用 jieba 进行中文分词，如果未安装则使用正则表达式进行简单分词。

        Args:
            text: 待分词的文本

        Returns:
            分词结果列表
        """
        if not text:
            return []
        # 优先使用 jieba 进行中文分词（如果已安装）
        if jieba is not None:
            return list(jieba.cut(text, cut_all=False))
        # 回退到简单的正则表达式分词（适用于英文和简单场景）
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return [t for t in tokens if t.strip()]

    def _char_width(self, block: _LinearBlock) -> float:
        """计算字符平均宽度。"""
        if not block.bbox or not block.block.text:
            return 0.0
        text_len = len(block.block.text)
        if text_len == 0:
            return 0.0
        width = block.bbox[2] - block.bbox[0]
        return width / text_len

    def _height(self, block: _LinearBlock) -> float:
        """计算块高度。"""
        if not block.bbox:
            return 0.0
        return block.bbox[3] - block.bbox[1]

    def _x_dis(self, a: _LinearBlock, b: _LinearBlock) -> float:
        """计算两个块的水平距离。"""
        if not a.bbox or not b.bbox:
            return float('inf')
        a_x0, _, a_x1, _ = a.bbox
        b_x0, _, b_x1, _ = b.bbox
        return min(abs(a_x1 - b_x0), abs(a_x0 - b_x1), abs(a_x0 + a_x1 - b_x0 - b_x1) / 2)

    def _y_dis(self, a: _LinearBlock, b: _LinearBlock) -> float:
        """计算两个块的垂直距离（中心点）。"""
        if not a.bbox or not b.bbox:
            return float('inf')
        a_y0, _, _, a_y1 = a.bbox
        b_y0, _, _, b_y1 = b.bbox
        a_center = (a_y0 + a_y1) / 2
        b_center = (b_y0 + b_y1) / 2
        return b_center - a_center

    def _extract_concat_features(
            self,
            up: _LinearBlock,
            down: _LinearBlock,
            mean_height: float,
    ) -> Optional[List[float]]:
        """提取段落拼接特征，用于 XGBoost 模型预测。

        目的：判断两个相邻的段落块是否应该合并为一个段落。
        提取 30+ 个特征，包括几何特征（位置、距离）、文本特征（标点、编号）、
        语义特征（分词、相似度）等，输入到 XGBoost 模型进行二分类决策。

        参考 RAGFlow 的 _updown_concat_features 实现，保持特征维度一致。

        Args:
            up: 上方的段落块
            down: 下方的段落块
            mean_height: 平均块高度（用于归一化）

        Returns:
            特征向量列表（30+ 维），如果无法提取则返回 None
        """
        if not up.block.text or not down.block.text:
            return None

        # 清理文本，去除首尾空白
        up_text = up.block.text.strip()
        down_text = down.block.text.strip()
        if not up_text or not down_text:
            return None

        # 计算几何特征：字符宽度、块高度、垂直距离
        # 这些特征用于判断两个块在视觉上是否连续
        w = max(self._char_width(up), self._char_width(down))  # 最大字符宽度
        h = max(self._height(up), self._height(down))  # 最大块高度
        y_dis = self._y_dis(up, down)  # 垂直距离（中心点之间）
        LEN = 6  # 用于分词的特征窗口大小

        # 分词特征：提取上下块边界处的分词信息
        # 用于判断文本在语义上是否连续（例如：单词被拆分到两行）
        tks_down = self._simple_tokenize(down_text[:LEN])  # 下方块的前几个字符的分词
        tks_up = self._simple_tokenize(up_text[-LEN:])  # 上方块的后几个字符的分词

        # 拼接文本：将上下块的边界文本拼接，检查是否需要空格连接
        # 例如："word" + "next" 需要空格，但 "word" + "，" 不需要
        connector = " " if re.match(r"[a-zA-Z0-9]+", up_text[-1] + down_text[0]) else ""
        tks_all_text = up_text[-LEN:].strip() + connector + down_text[:LEN].strip()
        tks_all = self._simple_tokenize(tks_all_text)  # 拼接后的分词结果

        # 计算 in_row（同一行的块数量，简化实现）
        # 原 RAGFlow 实现会统计同一行的块数量，这里简化处理
        in_row_up = 0
        in_row_down = 0

        # 构建特征向量（30+ 维），用于 XGBoost 模型预测
        fea = [
            # ========== 基础几何特征（3 维）==========
            # 特征 0: 是否在同一列（多列布局时，不同列的块不应合并）
            up.col_id == down.col_id if (up.col_id is not None and down.col_id is not None) else False,
            # 特征 1: 归一化的垂直距离（用于判断两个块是否在视觉上连续）
            y_dis / max(h, 0.000001),
            # 特征 2: 跨页数（跨页的块通常不应合并，除非是长段落）
            down.page - up.page,

            # ========== 布局类型特征（4 维，简化实现）==========
            # 特征 3-6: 布局类型（这里简化为都是文本类型）
            True,  # up["layout_type"] == down["layout_type"]
            True,  # up["layout_type"] == "text"
            True,  # down["layout_type"] == "text"
            False,  # up["layout_type"] == "table"
            False,  # down["layout_type"] == "table"

            # ========== 标点特征（7 维）==========
            # 特征 7: 上方块是否以句号、问号等结尾（表示句子结束，不应合并）
            bool(re.search(r"([。？！；!?;+)）]|[a-z]\.)$", up_text)),
            # 特征 8: 上方块是否以逗号、冒号等结尾（表示句子未结束，可能应合并）
            bool(re.search(r"[，：‘“、0-9（+-]$", up_text)),
            # 特征 9: 下方块是否以标点开头（可能表示新句子，不应合并）
            bool(re.search(r"(^.?[/,?;:\]，。；：’”？！》】）-])", down_text)),
            # 特征 10: 上方块是否是完整的括号内容（如 "(xxx)"，表示完整单元）
            bool(re.match(r"[\(（][^\(\)（）]+[）\)]$", up_text)),
            # 特征 11: 上方块是否以逗号结尾且未以句号结尾（句子未完成）
            bool(re.search(r"[，,][^。.]+$", up_text)),
            # 特征 12: 重复特征（保持与 RAGFlow 特征维度一致）
            bool(re.search(r"[，,][^。.]+$", up_text)),
            # 特征 13: 括号跨块匹配（上方块有左括号，下方块有右括号，应合并）
            bool(re.search(r"[\(（][^\)）]+$", up_text) and re.search(r"[\)）]", down_text)),

            # ========== 编号和格式特征（5 维）==========
            # 特征 14: 下方块是否是标题编号模式（如 "1.1"、"第一章"，不应合并）
            bool(self._match_heading_pattern(down_text)),
            # 特征 15: 下方块是否以大写字母开头（可能是新段落）
            bool(re.match(r"[A-Z]", down_text)),
            # 特征 16: 上方块是否以大写字母结尾（可能是单词的一部分）
            bool(re.match(r"[A-Z]", up_text[-1]) if up_text else False),
            # 特征 17: 上方块是否以小写字母或数字结尾（可能是单词的一部分，应合并）
            bool(re.match(r"[a-z0-9]", up_text[-1]) if up_text else False),
            # 特征 18: 下方块是否全是数字和符号（可能是数据行，不应合并）
            bool(re.match(r"[0-9.%,-]+$", down_text)),

            # ========== 文本相似度特征（1 维）==========
            # 特征 19: 文本边界是否相似（上下块边界字符相同，可能是同一单词被拆分）
            up_text.strip()[-2:] == down_text.strip()[:2] if len(up_text.strip()) > 1 and len(
                down_text.strip()) > 1 else False,

            # ========== 位置关系特征（3 维）==========
            # 特征 20: 上方块是否在下方块右侧（水平排列，不应合并）
            up.bbox[0] > down.bbox[2] if (up.bbox and down.bbox) else False,
            # 特征 21: 高度差异率（高度差异大可能不是同一段落）
            abs(self._height(up) - self._height(down)) / max(min(self._height(up), self._height(down)), 0.000001),
            # 特征 22: 归一化的水平距离（水平距离大可能不是同一段落）
            self._x_dis(up, down) / max(w, 0.000001),

            # ========== 文本长度特征（1 维）==========
            # 特征 23: 文本长度差异率（长度差异大可能不是同一段落）
            (len(up_text) - len(down_text)) / max(len(up_text), len(down_text), 1),

            # ========== 分词特征（3 维）==========
            # 特征 24: 拼接后分词数量变化（如果拼接后分词数增加，可能是两个独立句子）
            len(tks_all) - len(tks_up) - len(tks_down),
            # 特征 25: 上下块分词数量差异（用于判断文本复杂度）
            len(tks_down) - len(tks_up),
            # 特征 26: 边界分词是否相同（相同可能是同一单词被拆分）
            tks_down[-1] == tks_up[-1] if (tks_down and tks_up) else False,

            # ========== 行内特征（2 维，简化实现）==========
            # 特征 27: 最大行内块数（用于判断是否在同一行）
            max(in_row_down, in_row_up),
            # 特征 28: 行内块数差异（差异大可能不在同一行）
            abs(in_row_down - in_row_up),

            # ========== 名词特征（2 维，简化实现）==========
            # 特征 29: 下方块边界是否是单个长词（可能是标题或专有名词，不应合并）
            len(tks_down) == 1 and len(tks_down[0]) > 1 if tks_down else False,
            # 特征 30: 上方块边界是否是单个长词
            len(tks_up) == 1 and len(tks_up[0]) > 1 if tks_up else False,
        ]
        # 将所有特征转换为浮点数（布尔值转为 0.0/1.0）
        return [float(f) if isinstance(f, bool) else f for f in fea]

    def _should_merge_paragraphs(
            self,
            prev_block: _LinearBlock,
            next_block: _LinearBlock,
            body_font_size: float,
            mean_height: Optional[float] = None,
    ) -> bool:
        """判断两个段落是否应该合并。

        目的：判断两个相邻的文本块是否属于同一个段落，应该合并为一个块。
        这解决了 PDF 解析时一个段落被拆分成多个块的问题（例如：换行、分页）。

        策略：
        1. 优先使用 XGBoost 模型决策（如果模型已加载）
        2. 如果模型不可用，回退到基于规则的判断

        Args:
            prev_block: 前一个文本块
            next_block: 后一个文本块
            body_font_size: 正文字体大小（用于规则判断）
            mean_height: 平均块高度（用于模型特征，如果为 None 则使用 body_font_size）

        Returns:
            True 表示应该合并，False 表示不应合并
        """
        # ========== 基础条件检查 ==========
        # 只处理文本块
        if not prev_block.is_text or not next_block.is_text:
            return False
        # 必须有文本内容
        if not prev_block.block.text or not next_block.block.text:
            return False
        # 必须在同一页（跨页合并需要更严格的判断，这里暂不支持）
        if prev_block.page != next_block.page:
            return False
        # 必须在同一列（多列布局时，不同列的块不应合并）
        if (
                prev_block.col_id is not None
                and next_block.col_id is not None
                and prev_block.col_id != next_block.col_id
        ):
            return False
        # 如果任一块是标题，不应合并（标题应该独立成块）
        if self._match_heading_pattern(prev_block.block.text or ""):
            return False
        if self._match_heading_pattern(next_block.block.text or ""):
            return False

        # ========== 使用 XGBoost 模型决策（优先）==========
        if self.updown_concat_model is not None:
            try:
                # 如果未提供 mean_height，使用 body_font_size 作为默认值
                if mean_height is None:
                    mean_height = body_font_size
                # 提取特征向量（30+ 维）
                features = self._extract_concat_features(prev_block, next_block, mean_height)
                if features is not None:
                    # 构建 XGBoost 的 DMatrix（数据矩阵）
                    dmatrix = xgb.DMatrix([features])
                    # 模型预测：输出 0-1 之间的概率值
                    prediction = self.updown_concat_model.predict(dmatrix)[0]
                    # 模型输出 > 0.5 表示应该拼接（二分类阈值）
                    if prediction > 0.5:
                        return True
                    else:
                        return False
            except Exception as e:
                logger.debug("模型预测失败，回退到规则判断: {}", e)

        # ========== 规则判断（回退方案）==========
        # 如果上方块以句号、问号等结尾，表示句子结束，不应合并
        prev_text = prev_block.block.text.rstrip()
        if not prev_text:
            return False
        if prev_text[-1] in "。！？!?;；:：,，、)）」』":
            return False

        # 如果没有坐标信息，默认合并（保守策略）
        if prev_block.bbox is None or next_block.bbox is None:
            return True

        # 提取边界框坐标
        x0_p, y0_p, x1_p, y1_p = prev_block.bbox  # 上方块
        x0_n, y0_n, x1_n, y1_n = next_block.bbox  # 下方块

        # 计算垂直间距（下方块顶部 - 上方块底部）
        h_gap = y0_n - y1_p
        if h_gap < 0:
            h_gap = 0.0  # 如果重叠，间距为 0

        # 计算行高（取两个块高度的最大值，或使用正文字体大小）
        line_height = max(y1_p - y0_p, y1_n - y0_n, body_font_size * 1.0)
        # 如果垂直间距超过 1.5 倍行高，说明两个块之间有很大空白，不应合并
        if h_gap > line_height * 1.5:
            return False

        # 计算水平对齐度
        width_p = x1_p - x0_p  # 上方块宽度
        width_n = x1_n - x0_n  # 下方块宽度
        if width_p <= 0 or width_n <= 0:
            return True  # 宽度异常，默认合并

        # 计算左边界差异（用于判断是否对齐）
        left_diff = abs(x0_p - x0_n)
        # 左边界对齐容差：至少 5px，或两个块中较小宽度的 10%
        left_threshold = max(5.0, 0.1 * min(width_p, width_n))
        # 如果左边界差异超过容差，说明两个块不对齐，不应合并
        if left_diff > left_threshold:
            return False

        # 通过所有检查，应该合并
        return True

    def _merge_paragraphs(
            self,
            linear_blocks: List[_LinearBlock],
            body_font_size: float,
    ) -> List[_LinearBlock]:
        """合并应该连接的段落块。

        遍历排序后的块列表，使用 XGBoost 模型或规则判断
        相邻的两个段落是否应该合并为一个段落。

        Args:
            linear_blocks: 待合并的块列表
            body_font_size: 正文字体大小（用于规则判断）

        Returns:
            合并后的块列表
        """
        if not linear_blocks:
            return []
        sorted_blocks = self._sort_linear_blocks(linear_blocks)
        merged: List[_LinearBlock] = []
        current: Optional[_LinearBlock] = None
        for lb in sorted_blocks:
            if not lb.is_text or not lb.block.text:
                if current is not None:
                    merged.append(current)
                    current = None
                merged.append(lb)
                continue
            if current is None:
                current = lb
                continue
            # 计算当前页的平均高度（用于模型特征）
            mean_height = body_font_size
            if current.bbox and lb.bbox:
                mean_height = (self._height(current) + self._height(lb)) / 2

            # 判断是否应该合并
            if not self._should_merge_paragraphs(current, lb, body_font_size, mean_height):
                # 不应合并，将当前块加入结果，开始处理下一个块
                merged.append(current)
                current = lb
                continue

            # ========== 执行合并操作 ==========
            # 合并文本：去除换行符，在需要时添加空格
            new_text = (current.block.text or "").rstrip("\n")  # 去除尾部换行
            append_text = (lb.block.text or "").lstrip("\n")  # 去除首部换行
            # 如果上方块不以空格结尾，添加空格（避免单词粘连）
            if new_text and not new_text.endswith(" "):
                new_text = new_text + " "
            new_text = new_text + append_text

            # 合并边界框：取两个块的最小外接矩形
            x0_list = []  # 所有左边界
            y0_list = []  # 所有上边界
            x1_list = []  # 所有右边界
            y1_list = []  # 所有下边界
            if current.bbox is not None:
                x0_c, y0_c, x1_c, y1_c = current.bbox
                x0_list.append(x0_c)
                y0_list.append(y0_c)
                x1_list.append(x1_c)
                y1_list.append(y1_c)
            if lb.bbox is not None:
                x0_n, y0_n, x1_n, y1_n = lb.bbox
                x0_list.append(x0_n)
                y0_list.append(y0_n)
                x1_list.append(x1_n)
                y1_list.append(y1_n)
            # 计算合并后的边界框（最小外接矩形）
            if x0_list:
                new_bbox = (
                    min(x0_list),  # 最左边界
                    min(y0_list),  # 最上边界
                    max(x1_list),  # 最右边界
                    max(y1_list),  # 最下边界
                )
            else:
                new_bbox = None

            # 合并后的字体大小取较大值（保留重要格式信息）
            new_font_size = max(current.font_size, lb.font_size)

            # 更新坐标信息
            coords = current.block.coordinates
            if new_bbox is not None:
                coords = Coordinates(
                    page=current.page,
                    x=new_bbox[0],
                    y=new_bbox[1],
                    w=new_bbox[2] - new_bbox[0],  # 宽度
                    h=new_bbox[3] - new_bbox[1],  # 高度
                )
            new_block = Block(
                block_id=current.block.block_id,
                type=current.block.type,
                order=current.block.order,
                text=new_text,
                html=current.block.html,
                ocr_text=current.block.ocr_text,
                table=current.block.table,
                coordinates=coords,
                styles=current.block.styles,
                metadata=current.block.metadata,
            )
            current = _LinearBlock(
                block=new_block,
                page=current.page,
                bbox=new_bbox,
                font_size=new_font_size,
                is_text=True,
                global_order=current.global_order,
                col_id=current.col_id,
            )
        if current is not None:
            merged.append(current)
        return merged

    def _build_full_text(self, linear_blocks: List[_LinearBlock]) -> str:
        """构建文档的完整文本内容。

        将所有文本块按阅读顺序拼接成完整文本，包括段落文本、OCR文本和表格内容。

        Args:
            linear_blocks: 内容块列表

        Returns:
            完整的文档文本（用换行符分隔各块）
        """
        parts: List[str] = []
        for lb in self._sort_linear_blocks(linear_blocks):
            # 添加段落文本
            if lb.is_text and lb.block.text:
                parts.append(lb.block.text)
            # 添加图片OCR文本
            elif lb.block.type == BlockType.IMAGE and lb.block.ocr_text:
                parts.append(lb.block.ocr_text)
            # 添加表格文本（将表格转换为文本格式）
            elif lb.block.type == BlockType.TABLE and lb.block.table:
                table_text = []
                for row in lb.block.table.rows:
                    table_text.append("\t".join(str(cell) for cell in row))
                parts.append("\n".join(table_text))
        return "\n\n".join(parts)

    _BULLET_CHARS = set("•●◦▪▫■□◆▶➤▸▹-·*➢⚫")

    def _detect_heading_level(
            self,
            block: _LinearBlock,
            body_font_size: float,
            levels_by_size: dict[float, int],
    ) -> Optional[int]:
        """根据字体、编号和项目符号模式判断标题层级。

        目的：识别文本块是否为标题，并确定其层级（H1/H2/H3）。
        用于构建章节树，将标题作为 SectionNode，普通文本作为 Block。

        判断优先级：
        1. 字体大小匹配（最可靠）
        2. 编号模式匹配（如 "1.1"、"第一章"）
        3. 字体大小 + 文本长度（大字体 + 短文本 = 标题）
        4. 项目符号（如 "•"、"●"）
        5. 列位置 + 字体大小（左列 + 大字体 = 可能是标题）

        Args:
            block: 待判断的文本块
            body_font_size: 正文字体大小（用于对比）
            levels_by_size: 字体大小到标题层级的映射

        Returns:
            标题层级（1=H1, 2=H2, 3=H3），如果不是标题则返回 None
        """
        text = (block.block.text or "").strip()
        if not text:
            return None

        font_size = block.font_size
        # 字体大小容差：至少 0.6pt，或正文字体大小的 8%
        # 用于处理字体大小的小幅波动（例如：12.0pt vs 12.1pt 视为相同）
        tolerance = max(0.6, body_font_size * 0.08)

        # 方法 1: 根据字体大小匹配（从大到小检查，优先匹配大字体）
        for size in sorted(levels_by_size.keys(), reverse=True):
            level = levels_by_size[size]
            # 如果字体大小匹配且文本长度合理（标题通常不会太长），判定为标题
            if abs(font_size - size) <= tolerance and len(text) <= 120:
                return level

        # 方法 2: 根据编号模式匹配（如 "1.1"、"第一章"）
        numbered_level = self._match_heading_pattern(text)
        if numbered_level:
            return numbered_level

        # 方法 3: 大字体 + 短文本 = 可能是 H1 标题
        # 例如：文档标题、章节标题通常字体大、文本短
        if len(text) <= 40 and font_size >= body_font_size * 1.25:
            return 1

        # 方法 4: 项目符号开头 + 短文本 = 可能是 H3 标题（列表项）
        if text and text[0] in self._BULLET_CHARS and len(text) <= 60:
            return 3

        # 方法 5: 左列 + 大字体 + 短文本 = 可能是 H2 标题
        # 多列布局时，标题通常在左列
        if block.col_id == 0 and len(text) <= 25 and font_size >= body_font_size * 1.15:
            return 2

        # 不满足任何条件，不是标题
        return None

    def _semantic_chunking(self, sections: List[SectionNode], filename: str, content_type: str) -> List[Chunk]:
        """根据语义相关性对章节内容进行分块，生成与SectionNode平级的Chunk结构。

        Args:
            sections: 原始章节树
            filename: 文件名
            content_type: 文件类型 MIME

        Returns:
            生成的Chunk列表
        """
        chunks: List[Chunk] = []
        chunk_order = 1
        
        # 递归处理所有章节
        def process_section(section: SectionNode, parent_headings: List[str] = []):
            nonlocal chunks, chunk_order
            
            # 构建当前章节的标题链
            current_headings = parent_headings + [section.title]
            
            # 收集章节内的所有内容块
            all_blocks = []
            
            # 递归收集子章节的内容块
            def collect_blocks(sec: SectionNode):
                all_blocks.extend(sec.content_blocks)
                for child in sec.children:
                    collect_blocks(child)
            
            collect_blocks(section)
            
            if not all_blocks:
                return
            
            # 计算章节的页码范围
            page_start = min(block.coordinates.page for block in all_blocks if block.coordinates)
            page_end = max(block.coordinates.page for block in all_blocks if block.coordinates)
            
            # 1. 生成整节的Chunk
            section_content = ""
            section_html = ""
            section_block_ids = []
            
            for block in all_blocks:
                section_block_ids.append(block.block_id)
                if block.text:
                    section_content += block.text + "\n\n"
                if block.ocr_text:
                    section_content += block.ocr_text + "\n\n"
                if block.html:
                    section_html += block.html + "\n"
            
            section_content = section_content.strip()
            section_html = section_html.strip()
            
            if section_content:
                section_chunk = Chunk(
                    chunk_id=f"chunk-sec-{section.section_id}",
                    doc_id=filename,
                    section_id=section.section_id,
                    block_ids=section_block_ids,
                    chunk_type=ChunkType.SECTION_CHUNK,
                    order=chunk_order,
                    content=section_content,
                    raw_html=section_html,
                    token_count=len(section_content),
                    meta=ChunkMeta(
                        title=section.title,
                        headings=current_headings,
                        file_name=filename,
                        file_type=content_type,
                        page_start=page_start,
                        page_end=page_end
                    )
                )
                chunks.append(section_chunk)
                chunk_order += 1
            
            # 2. 生成段落级别的Chunk
            current_paragraph_blocks = []
            for block in all_blocks:
                if block.type == BlockType.PARAGRAPH:
                    current_paragraph_blocks.append(block)
                else:
                    # 非段落块，生成当前段落Chunk
                    if current_paragraph_blocks:
                        self._create_paragraph_chunk(
                            current_paragraph_blocks, 
                            section, 
                            current_headings, 
                            filename, 
                            content_type, 
                            chunks, 
                            chunk_order
                        )
                        chunk_order += 1
                        current_paragraph_blocks = []
                    
                    # 为非段落块生成单独的Chunk
                    self._create_non_paragraph_chunk(
                        block, 
                        section, 
                        current_headings, 
                        filename, 
                        content_type, 
                        chunks, 
                        chunk_order
                    )
                    chunk_order += 1
            
            # 处理最后一组段落块
            if current_paragraph_blocks:
                self._create_paragraph_chunk(
                    current_paragraph_blocks, 
                    section, 
                    current_headings, 
                    filename, 
                    content_type, 
                    chunks, 
                    chunk_order
                )
                chunk_order += 1
            
            # 递归处理子章节
            for child in section.children:
                process_section(child, current_headings)
        
        # 开始处理根章节
        for section in sections:
            process_section(section)
        
        return chunks
    
    def _create_paragraph_chunk(self, blocks: List[Block], section: SectionNode, headings: List[str], 
                               filename: str, content_type: str, chunks: List[Chunk], order: int):
        """创建段落级别的Chunk。
        
        Args:
            blocks: 段落块列表
            section: 所属章节
            headings: 标题链
            filename: 文件名
            content_type: 文件类型
            chunks: Chunk列表，用于添加生成的Chunk
            order: Chunk顺序
        """
        content = ""
        html = ""
        block_ids = []
        
        for block in blocks:
            block_ids.append(block.block_id)
            if block.text:
                content += block.text + "\n\n"
            if block.html:
                html += block.html + "\n"
        
        content = content.strip()
        html = html.strip()
        
        if not content:
            return
        
        # 计算页码范围
        page_start = min(block.coordinates.page for block in blocks if block.coordinates)
        page_end = max(block.coordinates.page for block in blocks if block.coordinates)
        
        chunk = Chunk(
            chunk_id=f"chunk-para-{section.section_id}-{order}",
            doc_id=filename,
            section_id=section.section_id,
            block_ids=block_ids,
            chunk_type=ChunkType.PARAGRAPH_CHUNK,
            order=order,
            content=content,
            raw_html=html,
            token_count=len(content),
            meta=ChunkMeta(
                title=section.title,
                headings=headings,
                file_name=filename,
                file_type=content_type,
                page_start=page_start,
                page_end=page_end
            )
        )
        chunks.append(chunk)
    
    def _create_non_paragraph_chunk(self, block: Block, section: SectionNode, headings: List[str], 
                                   filename: str, content_type: str, chunks: List[Chunk], order: int):
        """创建非段落级别的Chunk（图片、表格等）。
        
        Args:
            block: 非段落块
            section: 所属章节
            headings: 标题链
            filename: 文件名
            content_type: 文件类型
            chunks: Chunk列表，用于添加生成的Chunk
            order: Chunk顺序
        """
        content = ""
        html = block.html or ""
        
        if block.type == BlockType.IMAGE:
            content = block.ocr_text or "[图片]"
        elif block.type == BlockType.TABLE:
            # 将表格转换为文本格式
            if block.table:
                for row in block.table.rows:
                    content += "\t".join(str(cell) for cell in row) + "\n"
        
        content = content.strip()
        
        # 计算页码范围
        page_start = block.coordinates.page if block.coordinates else 1
        page_end = block.coordinates.page if block.coordinates else 1
        
        chunk = Chunk(
            chunk_id=f"chunk-{block.type.value}-{block.block_id}",
            doc_id=filename,
            section_id=section.section_id,
            block_ids=[block.block_id],
            chunk_type=ChunkType.PARAGRAPH_CHUNK,
            order=order,
            content=content,
            raw_html=html,
            token_count=len(content),
            meta=ChunkMeta(
                title=section.title,
                headings=headings,
                file_name=filename,
                file_type=content_type,
                page_start=page_start,
                page_end=page_end
            )
        )
        chunks.append(chunk)

    def _match_heading_pattern(self, text: str) -> Optional[int]:
        """根据常见标题编号模式识别标题层级。

        目的：通过正则表达式匹配常见的标题编号格式，提高标题识别准确率。
        这些模式通常比字体大小更可靠，因为编号格式是结构化的。

        支持的格式：
        - 中文：第一章、第一节、一、 （一）、1.1、1）等
        - 英文：A.、I.、1.1 等
        - 项目符号：•、● 等

        Args:
            text: 待匹配的文本

        Returns:
            标题层级（1=H1, 2=H2, 3=H3），如果不匹配则返回 None
        """
        stripped = text.strip()
        if not stripped:
            return None

        # ========== 固定模式匹配（优先级从高到低）==========
        pattern_map: list[tuple[str, int]] = [
            # H1: 中文章节（如 "第一章"、"第二章"）
            (r"^第[零一二三四五六七八九十百千]+章", 1),
            # H2: 中文节/条（如 "第一节"、"第一条"）
            (r"^第[零一二三四五六七八九十百千]+[节条]", 2),
            # H2: 中文数字 + 顿号/空格（如 "一、"、"二 "）
            (r"^[零一二三四五六七八九十百]+[、\s]", 2),
            # H3: 中文括号数字（如 "（一）"、"（二）"）
            (r"^[（(][零一二三四五六七八九十]+[）)]", 3),
            # H3: 英文大写字母 + 点/括号（如 "A."、"B)"）
            (r"^[A-Z][\.\)]\s", 3),
            # H2: 罗马数字（如 "I."、"II."、"III."）
            (r"^[IVXLCDM]+\.", 2),
        ]
        # 按顺序匹配，匹配到第一个就返回
        for patt, level in pattern_map:
            if re.match(patt, stripped):
                return level

        # ========== 数字编号模式（动态层级）==========
        # 匹配数字编号（如 "1."、"1.1"、"1.1.1"、"1.1.1.1"）
        # 层级由点的数量决定：1. -> H1, 1.1 -> H2, 1.1.1 -> H3
        m = re.match(r"^(\d+(?:\.\d+){0,3})[\.、．\s]", stripped)
        if m:
            # 计算点的数量（即层级深度）
            # 例如："1.1.1" -> ["1", "1", "1"] -> 3 层
            depth = len(m.group(1).split("."))
            # 最多支持 3 级标题
            return min(3, depth)

        # ========== 其他模式 ==========
        # H3: 数字 + 右括号（如 "1）"、"2）"）
        if re.match(r"^\d+[）)]", stripped):
            return 3

        # H3: 项目符号开头（如 "•"、"●"）
        if stripped[:1] in self._BULLET_CHARS:
            return 3

        # 不匹配任何模式
        return None

    def _open_pdf(self, content: bytes):
        """打开 PDF，使用 PyMuPDF（pymupdf）。

        Args:
            content: PDF 文件的二进制内容

        Returns:
            PyMuPDF 的 Document 对象

        Raises:
            RuntimeError: 如果 PDF 打开失败
        """
        try:
            return fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            logger.error("打开 PDF 失败: {}", e)
            raise
