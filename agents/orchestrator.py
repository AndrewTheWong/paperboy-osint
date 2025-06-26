"""
StraitWatch Orchestrator Agent

Main orchestrator that coordinates all background agents and manages scheduling.
"""

import asyncio
import schedule
import logging
from datetime import datetime
from typing import Dict, Any, List
import json

from .base_agent import BaseAgent

class OrchestratorAgent(BaseAgent):
    """Main orchestrator for all StraitWatch agents"""
    
    def __init__(self):
        super().__init__("orchestrator")
        
        # Import agents dynamically to avoid circular imports
        self.agents = {}
        self.schedules = {}
        self.initialize_agents()
        
    def initialize_agents(self):
        """Initialize all managed agents"""
        try:
            from .ingestion_agent import IngestionAgent
            self.agents['ingestion'] = IngestionAgent()
        except ImportError as e:
            self.logger.warning(f"Could not import IngestionAgent: {e}")
            
        try:
            from .nlp_agent import NLPAgent
            self.agents['nlp'] = NLPAgent()
        except ImportError as e:
            self.logger.warning(f"Could not import NLPAgent: {e}")
            
        # Additional agents can be added as they're implemented
        self.logger.info(f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")
    
    def setup_schedules(self):
        """Setup automatic schedules for all agents"""
        
        # Ingestion every hour
        if 'ingestion' in self.agents:
            schedule.every().hour.do(self.run_agent_safe, 'ingestion')
            self.logger.info("Scheduled ingestion agent: every hour")
        
        # NLP processing every 2 hours
        if 'nlp' in self.agents:
            schedule.every(2).hours.do(self.run_agent_safe, 'nlp')
            self.logger.info("Scheduled NLP agent: every 2 hours")
        
        # Time series building every 6 hours
        if 'timeseries' in self.agents:
            schedule.every(6).hours.do(self.run_agent_safe, 'timeseries')
            self.logger.info("Scheduled timeseries agent: every 6 hours")
        
        # Forecasting daily at midnight
        if 'forecasting' in self.agents:
            schedule.every().day.at("00:00").do(self.run_agent_safe, 'forecasting')
            self.logger.info("Scheduled forecasting agent: daily at 00:00")
        
        # Reports daily at 6 AM
        if 'report' in self.agents:
            schedule.every().day.at("06:00").do(self.run_agent_safe, 'report')
            self.logger.info("Scheduled report agent: daily at 06:00")
    
    async def run(self) -> Dict[str, Any]:
        """Main orchestrator run - health check and coordination"""
        
        results = {}
        
        # Health check all agents
        for agent_name, agent in self.agents.items():
            try:
                health = await agent.health_check()
                status = agent.get_status()
                results[agent_name] = {
                    "healthy": health,
                    "status": status
                }
            except Exception as e:
                results[agent_name] = {
                    "healthy": False,
                    "error": str(e)
                }
        
        # Check if any agents need immediate attention
        unhealthy_agents = [name for name, result in results.items() 
                          if not result.get("healthy", False)]
        
        if unhealthy_agents:
            self.logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
        
        return {
            "agent_health": results,
            "unhealthy_count": len(unhealthy_agents),
            "total_agents": len(self.agents)
        }
    
    def run_agent_safe(self, agent_name: str):
        """Safely run an agent (synchronous wrapper for scheduler)"""
        if agent_name not in self.agents:
            self.logger.error(f"Agent {agent_name} not found")
            return
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                self.agents[agent_name].safe_run()
            )
            loop.close()
            
            self.logger.info(f"Agent {agent_name} completed: {result}")
            
        except Exception as e:
            self.logger.error(f"Error running agent {agent_name}: {e}")
    
    async def run_agent_async(self, agent_name: str) -> Dict[str, Any]:
        """Run an agent asynchronously"""
        if agent_name not in self.agents:
            return {"error": f"Agent {agent_name} not found"}
        
        return await self.agents[agent_name].safe_run()
    
    async def run_all_agents(self) -> Dict[str, Any]:
        """Run all agents once"""
        results = {}
        
        for agent_name in self.agents:
            self.logger.info(f"Running agent: {agent_name}")
            results[agent_name] = await self.run_agent_async(agent_name)
            
            # Small delay between agents
            await asyncio.sleep(5)
        
        return results
    
    async def start_scheduler(self):
        """Start the automatic scheduler"""
        self.setup_schedules()
        
        self.logger.info("Starting StraitWatch scheduler...")
        
        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        status = {}
        
        for agent_name, agent in self.agents.items():
            status[agent_name] = agent.get_status()
        
        return {
            "system_status": "operational" if all(
                s.get("status") in ["healthy", "degraded"] 
                for s in status.values()
            ) else "critical",
            "agents": status,
            "scheduler_running": True,
            "timestamp": datetime.now().isoformat()
        }

# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="StraitWatch Orchestrator")
    parser.add_argument("--mode", choices=["schedule", "run-once", "status"], 
                       default="schedule", help="Orchestrator mode")
    parser.add_argument("--agent", type=str, help="Run specific agent only")
    
    args = parser.parse_args()
    
    async def main():
        orchestrator = OrchestratorAgent()
        
        if args.mode == "schedule":
            await orchestrator.start_scheduler()
        elif args.mode == "run-once":
            if args.agent:
                result = await orchestrator.run_agent_async(args.agent)
                print(f"Agent {args.agent} result: {json.dumps(result, indent=2)}")
            else:
                results = await orchestrator.run_all_agents()
                print(f"All agents results: {json.dumps(results, indent=2)}")
        elif args.mode == "status":
            status = orchestrator.get_system_status()
            print(json.dumps(status, indent=2))
    
    asyncio.run(main())