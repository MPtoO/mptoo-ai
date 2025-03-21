from pydantic import BaseModel, AnyHttpUrl
from typing import List, Dict, Any, Optional
from enum import Enum


class CrawlStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CrawlRequest(BaseModel):
    """Modèle pour une demande de crawl"""
    url: AnyHttpUrl
    max_depth: Optional[int] = 2
    max_pages: Optional[int] = 100
    respect_robots_txt: Optional[bool] = True
    custom_headers: Optional[Dict[str, str]] = None
    priority: Optional[int] = 1  # 1-10, 10 étant la plus haute priorité


class CrawlPageResult(BaseModel):
    """Résultat du crawl d'une page"""
    url: str
    title: Optional[str] = None
    content_snippet: Optional[str] = None
    status_code: int
    crawl_time: str  # ISO datetime
    links_found: int
    size: Optional[int] = None  # taille en octets


class CrawlResult(BaseModel):
    """Modèle pour un résultat de crawl"""
    target_id: str
    url: str
    status: CrawlStatus
    start_time: Optional[str] = None  # ISO datetime
    end_time: Optional[str] = None  # ISO datetime
    pages_crawled: Optional[int] = None
    pages_success: Optional[int] = None
    pages_failed: Optional[int] = None
    message: Optional[str] = None
    results: Optional[List[CrawlPageResult]] = None 