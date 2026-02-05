"""
提示词加载器

特性：
1. 启动时一次性加载并缓存，运行时不读磁盘
2. 支持热更新（reload方法）
3. 变量模板替换（{{VAR}}语法）
4. 类型安全的访问方式
5. 单例模式，全局唯一实例
"""

import re
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import yaml
from loguru import logger


@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    version: str
    description: str
    content: str
    variables: List[str] = field(default_factory=list)
    
    def render(self, **kwargs) -> str:
        """
        渲染模板，替换变量
        
        Args:
            **kwargs: 模板变量，key对应{{KEY}}
            
        Returns:
            渲染后的字符串
        """
        result = self.content
        for key, value in kwargs.items():
            placeholder = "{{" + key + "}}"
            result = result.replace(placeholder, str(value))
        return result
    
    def get_missing_variables(self, **kwargs) -> List[str]:
        """检查缺少的必要变量"""
        provided = set(kwargs.keys())
        required = set(self.variables)
        return list(required - provided)


class PromptLoader:
    """
    提示词加载器
    
    使用方式：
        # 启动时加载
        loader = get_prompt_loader()
        loader.load_all()
        
        # 使用时获取
        prompt = loader.get("scenario/phase_planning")
        rendered = prompt.render(EQUIPMENT_LIST="...", USER_REQUEST="...")
        
        # 或直接渲染
        text = loader.render("scenario/phase_planning", EQUIPMENT_LIST="...")
    """
    
    _instance: Optional["PromptLoader"] = None
    _initialized: bool = False
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._cache: Dict[str, PromptTemplate] = {}
        self._base_path = Path(__file__).parent
        self._initialized = True
    
    def load_all(self, force: bool = False) -> int:
        """
        加载所有提示词文件到缓存
        
        Args:
            force: 强制重新加载（即使已加载）
            
        Returns:
            加载的文件数量
        """
        if self._cache and not force:
            logger.debug("提示词已加载，跳过（共{}个）", len(self._cache))
            return len(self._cache)
        
        if force:
            self._cache.clear()
        
        count = 0
        for md_file in self._base_path.rglob("*.md"):
            try:
                template = self._parse_file(md_file)
                key = self._file_to_key(md_file)
                self._cache[key] = template
                count += 1
                logger.debug("加载提示词: {}", key)
            except Exception as e:
                logger.error("加载提示词失败: {} -> {}", md_file, e)
        
        logger.info("提示词加载完成，共 {} 个", count)
        return count
    
    def get(self, key: str) -> Optional[PromptTemplate]:
        """
        获取提示词模板
        
        Args:
            key: 提示词key，如 "scenario/phase_planning"
            
        Returns:
            PromptTemplate 或 None
        """
        template = self._cache.get(key)
        if template is None:
            logger.warning("提示词不存在: {}", key)
        return template
    
    def render(self, key: str, **kwargs) -> str:
        """
        直接获取并渲染提示词
        
        Args:
            key: 提示词key
            **kwargs: 模板变量
            
        Returns:
            渲染后的字符串
            
        Raises:
            KeyError: 提示词不存在
        """
        template = self.get(key)
        if not template:
            raise KeyError(f"提示词不存在: {key}")
        return template.render(**kwargs)
    
    def exists(self, key: str) -> bool:
        """检查提示词是否存在"""
        return key in self._cache
    
    def _file_to_key(self, file_path: Path) -> str:
        """将文件路径转换为key"""
        relative = file_path.relative_to(self._base_path)
        # 移除.md后缀，使用/分隔
        return str(relative.with_suffix("")).replace("\\", "/")
    
    def _parse_file(self, file_path: Path) -> PromptTemplate:
        """
        解析Markdown文件
        
        支持格式：
        ---
        name: xxx
        version: "1.0"
        description: xxx
        variables:
          - VAR1
          - VAR2
        ---
        
        实际提示词内容...
        """
        content = file_path.read_text(encoding="utf-8")
        
        # 解析YAML Frontmatter
        frontmatter: Dict[str, Any] = {}
        body = content
        
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                except yaml.YAMLError as e:
                    logger.warning("解析frontmatter失败: {} -> {}", file_path, e)
                body = parts[2].strip()
        
        return PromptTemplate(
            name=frontmatter.get("name", file_path.stem),
            version=str(frontmatter.get("version", "1.0")),
            description=frontmatter.get("description", ""),
            content=body,
            variables=frontmatter.get("variables", []),
        )
    
    def reload(self) -> int:
        """
        热更新：重新加载所有提示词
        
        Returns:
            重新加载的文件数量
        """
        logger.info("热更新提示词...")
        return self.load_all(force=True)
    
    def list_keys(self) -> List[str]:
        """列出所有提示词key"""
        return sorted(self._cache.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_count": len(self._cache),
            "keys": self.list_keys(),
            "base_path": str(self._base_path),
        }


# ============== 便捷访问函数 ==============

def get_prompt_loader() -> PromptLoader:
    """获取提示词加载器单例"""
    return PromptLoader()


def render_prompt(key: str, **kwargs) -> str:
    """
    便捷渲染函数
    
    Args:
        key: 提示词key
        **kwargs: 模板变量
        
    Returns:
        渲染后的字符串
    """
    return get_prompt_loader().render(key, **kwargs)


def load_prompts() -> int:
    """
    加载所有提示词（启动时调用）
    
    Returns:
        加载的数量
    """
    return get_prompt_loader().load_all()
