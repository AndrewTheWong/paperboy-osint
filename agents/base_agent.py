"""
Base Agent Class for StraitWatch Background Agents
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
from utils.supabase_client import get_supabase

class BaseAgent(ABC):
    """Base class for all StraitWatch background agents"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.name = name
        self.supabase = get_supabase()
        self.is_running = False
        self.last_run = None
        self.error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"straitwatch.{name}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create console handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Main execution method for the agent"""
        pass
    
    async def safe_run(self) -> Dict[str, Any]:
        """Safely execute the agent with error handling and logging"""
        self.logger.info(f"Starting {self.name} agent run")
        start_time = datetime.now()
        
        try:
            self.is_running = True
            result = await self.run()
            
            self.last_run = datetime.now()
            self.error_count = 0
            
            # Log success metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"{self.name} completed successfully in {duration:.2f}s")
            
            # Store run metrics
            await self.log_run_metrics(start_time, duration, result, success=True)
            
            return result
            
        except Exception as e:
            self.error_count += 1
            duration = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"{self.name} failed after {duration:.2f}s: {str(e)}")
            
            # Store error metrics
            await self.log_run_metrics(start_time, duration, {"error": str(e)}, success=False)
            
            # Re-raise if too many consecutive errors
            if self.error_count >= 5:
                self.logger.critical(f"{self.name} has failed 5 times consecutively")
                raise
            
            return {"success": False, "error": str(e)}
            
        finally:
            self.is_running = False
    
    async def log_run_metrics(self, start_time: datetime, duration: float, 
                            result: Dict[str, Any], success: bool):
        """Log agent run metrics to database"""
        try:
            metrics = {
                "agent_name": self.name,
                "start_time": start_time.isoformat(),
                "duration_seconds": duration,
                "success": success,
                "result_data": result,
                "error_count": self.error_count
            }
            
            self.supabase.table("agent_runs").insert(metrics).execute()
            
        except Exception as e:
            self.logger.warning(f"Failed to log metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "name": self.name,
            "is_running": self.is_running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "error_count": self.error_count,
            "status": "healthy" if self.error_count < 3 else "degraded" if self.error_count < 5 else "critical"
        }
    
    async def health_check(self) -> bool:
        """Perform a health check for this agent"""
        try:
            # Basic health check - can connect to database
            self.supabase.table("articles").select("count").limit(1).execute()
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False