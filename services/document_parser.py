from __future__ import annotations

from typing import Optional, List
from abc import ABC, abstractmethod
from loguru import logger

from models.parsed_document import ParsedDocument


class BaseDocumentParser(ABC):
    """文档解析器接口。负责将原始文件解析为 ParsedDocument 结构。"""

    @abstractmethod
    def supports(self, filename: str, content_type: Optional[str] = None) -> bool:
        ...

    @abstractmethod
    def parse(self, content: bytes, filename: str, content_type: Optional[str] = None) -> ParsedDocument:
        ...


class CompositeDocumentParser:
    """聚合解析器：根据文件类型委派给具体解析器实现。"""

    def __init__(self, parsers: Optional[List[BaseDocumentParser]] = None) -> None:
        self._parsers: List[BaseDocumentParser] = parsers or []

    def register(self, parser: BaseDocumentParser) -> None:
        self._parsers.append(parser)

    def parse(self, content: bytes, filename: str, content_type: Optional[str] = None) -> ParsedDocument:
        for parser in self._parsers:
            if parser.supports(filename, content_type):
                logger.info(f"使用解析器 {parser.__class__.__name__} 解析文件: {filename}")
                return parser.parse(content, filename, content_type)
        raise ValueError(f"不支持的文件类型: filename={filename}, content_type={content_type}")


_global_document_parser: Optional[CompositeDocumentParser] = None


def get_document_parser() -> CompositeDocumentParser:
    """获取全局文档解析器实例。"""
    global _global_document_parser
    if _global_document_parser is None:
        from services.pdf_parser import PdfDocumentParser  # 局部导入避免循环依赖

        parser = CompositeDocumentParser()
        parser.register(PdfDocumentParser())
        _global_document_parser = parser
        logger.info("全局 CompositeDocumentParser 初始化完成，已注册 PDF 解析器")
    return _global_document_parser
