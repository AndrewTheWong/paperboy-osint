"""
Orchestrator Agent for StraitWatch
Coordinates all background agents and manages their schedules
"""

import asyncio
import logging
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path
import json
import signal
import sys

from agents.base_agent import BaseAgent
from agents.article_ingest_agent import ArticleIngestAgent
from agents.nlp_agent import NLPAgent
from agents.timeseries_builder_agent import TimeSeriesBuilderAgent
from agents.forecasting_agent import ForecastingAgent
from utils.supabase_client import get_supabase

class OrchestratorAgent(BaseAgent):
    """Main orchestrator that coordinates all StraitWatch agents"""
    
    def __init__(self):
        super().__init__("orchestrator")
        self.agents = {}
        self.running = False
        self.schedules = {}
        
        # Initialize all agents
        self.initialize_agents()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def initialize_agents(self):
        """Initialize all background agents"""
        from .article_ingest_agent import ArticleIngestAgent
        from .tagging_agent import TaggingAgent
        from .timeseries_builder_agent import TimeSeriesBuilderAgent
        from .forecasting_agent import ForecastingAgent
        from .report_generator_agent import ReportGeneratorAgent
        
        self.agents = {
            'article_ingest': ArticleIngestAgent(),
            'tagging': TaggingAgent(),
            'timeseries_builder': TimeSeriesBuilderAgent(),
            'forecasting': ForecastingAgent(),
            'report_generator': ReportGeneratorAgent()
        }
        
        self.logger.info(f"Initialized {len(self.agents)} StraitWatch agents")
    
    async def run(self) -> Dict[str, Any]:
        """Main orchestrator run loop"""
        self.logger.info("Starting StraitWatch orchestrator")
        self.running = True
        
        # Setup schedules
        self.setup_schedules()
        
        # Run initial tasks
        await self.run_initial_tasks()
        
        # Start the main loop
        await self.run_scheduler_loop()
        
        return {"success": True, "message": "Orchestrator completed"}
    
    def setup_schedules(self):
        """Setup agent schedules"""
        # Article Ingestion: Every hour
        schedule.every().hour.do(self.run_agent, 'article_ingest')
        
        # NLP Tagging: Every 2 hours
        schedule.every(2).hours.do(self.run_agent, 'tagging')
        
        # Time Series Building: Every 6 hours
        schedule.every(6).hours.do(self.run_agent, 'timeseries_builder')
        
        # Forecasting: Daily at 05:00 UTC
        schedule.every().day.at("05:00").do(self.run_agent, 'forecasting')
        
        # Report Generation: Daily at 06:00 UTC
        schedule.every().day.at("06:00").do(self.run_agent, 'report_generator')
        
        # Health check: Every 30 minutes
        schedule.every(30).minutes.do(self.run_health_check)
        
        self.logger.info("StraitWatch agent schedules configured")
    
    async def run_initial_tasks(self):
        """Run initial tasks on startup"""
        self.logger.info("Running initial tasks")
        
        # Run time series builder first (needed for forecasting)
        await self.run_agent('timeseries_builder')
        
        # Run forecasting if it's been more than 24 hours
        last_forecast = await self.get_last_forecast_time()
        if last_forecast is None or (datetime.now() - last_forecast).days >= 1:
            await self.run_agent('forecasting')
        
        self.logger.info("Initial tasks completed")
    
    async def run_scheduler_loop(self):
        """Main scheduler loop"""
        self.logger.info("Starting scheduler loop")
        
        while self.running:
            try:
                # Run pending scheduled tasks
                schedule.run_pending()
                
                # Sleep for 1 minute
                await asyncio.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def run_agent(self, agent_name: str):
        """Run a specific agent"""
        if agent_name not in self.agents:
            self.logger.error(f"Unknown agent: {agent_name}")
            return
        
        self.logger.info(f"Running agent: {agent_name}")
        
        try:
            agent = self.agents[agent_name]
            result = await agent.safe_run()
            
            # Log result
            await self.log_agent_run(agent_name, result)
            
            # Check for critical errors
            if not result.get('success', False):
                self.logger.error(f"Agent {agent_name} failed: {result.get('error', 'Unknown error')}")
                
                # Trigger alert for critical agents
                if agent_name in ['article_ingest', 'tagging']:
                    await self.send_alert(f"Critical agent {agent_name} failed")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running agent {agent_name}: {e}")
            await self.log_agent_run(agent_name, {"success": False, "error": str(e)})
            return {"success": False, "error": str(e)}
    
    async def run_health_check(self):
        """Run health check on all agents"""
        self.logger.info("Running health check")
        
        health_status = {}
        for name, agent in self.agents.items():
            try:
                is_healthy = await agent.health_check()
                health_status[name] = {
                    "healthy": is_healthy,
                    "status": agent.get_status()
                }
            except Exception as e:
                health_status[name] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # Log health status
        await self.log_health_status(health_status)
        
        # Check overall system health
        healthy_agents = sum(1 for status in health_status.values() if status.get('healthy', False))
        total_agents = len(health_status)
        
        if healthy_agents < total_agents * 0.8:  # Less than 80% healthy
            await self.send_alert(f"System health degraded: {healthy_agents}/{total_agents} agents healthy")
        
        return health_status
    
    async def get_last_forecast_time(self) -> datetime:
        """Get the last time forecasting was run"""
        try:
            # Query the database for last forecast run
            supabase = get_supabase()
            result = supabase.table("agent_runs").select("start_time").eq("agent_name", "forecasting_agent").order("start_time", desc=True).limit(1).execute()
            
            if result.data:
                return datetime.fromisoformat(result.data[0]['start_time'])
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting last forecast time: {e}")
            return None
    
    async def log_agent_run(self, agent_name: str, result: Dict[str, Any]):
        """Log agent run results"""
        try:
            log_entry = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "result": result,
                "success": result.get('success', False)
            }
            
            # Save to database
            supabase = get_supabase()
            supabase.table("agent_runs").insert(log_entry).execute()
            
        except Exception as e:
            self.logger.error(f"Error logging agent run: {e}")
    
    async def log_health_status(self, health_status: Dict[str, Any]):
        """Log system health status"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "health_status": health_status,
                "overall_health": "healthy" if all(status.get('healthy', False) for status in health_status.values()) else "degraded"
            }
            
            # Save to database
            supabase = get_supabase()
            supabase.table("system_health").insert(log_entry).execute()
            
        except Exception as e:
            self.logger.error(f"Error logging health status: {e}")
    
    async def send_alert(self, message: str):
        """Send alert for critical issues"""
        try:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "message": message,
                "severity": "critical",
                "acknowledged": False
            }
            
            # Save to database
            supabase = get_supabase()
            supabase.table("alerts").insert(alert).execute()
            
            # Could also send email/SMS here
            self.logger.critical(f"ALERT: {message}")
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully")
        self.running = False
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("Shutting down orchestrator")
        self.running = False
        
        # Wait for any running tasks to complete
        await asyncio.sleep(5)
        
        self.logger.info("Orchestrator shutdown complete")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "running": self.running,
            "agents": {}
        }
        
        for name, agent in self.agents.items():
            status["agents"][name] = agent.get_status()
        
        return status

# Test function
async def test_orchestrator():
    """Test the orchestrator agent"""
    orchestrator = OrchestratorAgent()
    
    try:
        # Run for a short time to test
        await asyncio.wait_for(orchestrator.run(), timeout=300)  # 5 minutes
    except asyncio.TimeoutError:
        await orchestrator.shutdown()
    
    return orchestrator.get_system_status()

if __name__ == "__main__":
    # Run the orchestrator
    asyncio.run(test_orchestrator()) 