"""
Run a workflow using PocketFlow and Open Manus integration.

This script provides a command-line interface for running various integrated workflows
between PocketFlow and Open Manus, using the fixed implementation that avoids
circular dependencies and execution issues.
"""
import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

# Import agents
from app.agent.planning_mock import PlanningAgent
from app.agent.react import ReActAgent
from app.agent.planning import PlanningAgent as FullPlanningAgent

# Import tools
from app.tool.bash import Bash
from app.tool.google_search import GoogleSearch
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.python_execute import PythonExecute

# Import fixed PocketFlow integration
from app.pocketflow.fixed_orchestrator import WorkflowOrchestrator, CustomFlow
from app.pocketflow.fixed_adapters import ToolNode, AgentNode, PocketFlowNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleReActAgent(ReActAgent):
    """A simple ReAct agent for testing."""
    
    def __init__(self, name: str = "react_agent"):
        """Initialize a simple ReAct agent."""
        super().__init__(name=name)
        self.system_prompt = f"""
        You are a helpful assistant named {name}.
        """
        self.tools = []
    
    def add_tool(self, tool):
        """Add a tool to the agent."""
        self.tools.append(tool)
    
    async def think(self):
        """Think about the next step."""
        # Simple implementation for testing
        return False
    
    async def act(self):
        """Perform an action."""
        # Simple implementation for testing
        return "Action completed successfully."


class ExampleWorkflows:
    """Collection of example workflows using the fixed PocketFlow integration."""
    
    @staticmethod
    async def run_multi_agent_workflow(
        task: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a multi-agent workflow that coordinates planning, research, and code execution.
        
        This example creates a workflow with three agents:
        1. Planning agent - creates a plan for solving the task
        2. Research agent - gathers information needed for the task
        3. Execution agent - implements the solution based on research and planning
        
        Args:
            task: The task to complete
            context: Additional context for the task
            
        Returns:
            The results from the workflow execution
        """
        # Initialize the orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Create agents
        planning_agent = PlanningAgent(name="planner")
        research_agent = SimpleReActAgent(name="researcher")
        execution_agent = SimpleReActAgent(name="executor")
        
        # Create tools
        search_tool = GoogleSearch()
        bash_tool = Bash()
        editor_tool = StrReplaceEditor()
        
        # Add tools to agents
        research_agent.add_tool(search_tool)
        execution_agent.add_tool(bash_tool)
        execution_agent.add_tool(editor_tool)
        
        # Register tools and agents
        orchestrator.register_tool(search_tool)
        orchestrator.register_tool(bash_tool)
        orchestrator.register_tool(editor_tool)
        
        orchestrator.register_agent(planning_agent)
        orchestrator.register_agent(research_agent)
        orchestrator.register_agent(execution_agent)
        
        # Create a workflow
        workflow = orchestrator.create_agent_workflow(
            agents=[planning_agent, research_agent, execution_agent],
            connections={
                "planner": ["researcher"],
                "researcher": ["executor"]
            },
            flow_name="MultiAgentWorkflow"
        )
        
        # Run the workflow
        result = orchestrator.run_workflow(
            workflow_name="MultiAgentWorkflow",
            inputs={"task": task, "context": context or {}}
        )
        
        return result
    
    @staticmethod
    async def run_planning_with_parallel_execution(
        task: str,
        tools: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Run a workflow that uses planning to create a parallel execution plan.
        
        This example creates a workflow where a planning agent creates a plan with
        tasks that can be executed in parallel, and then executes them.
        
        Args:
            task: The task to complete
            tools: Optional list of tools to use for execution
            
        Returns:
            The results from the workflow execution
        """
        # Initialize the orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Create agents
        planning_agent = FullPlanningAgent(name="planner")
        execution_agent = SimpleReActAgent(name="executor")
        
        # Create tools if not provided
        if not tools:
            tools = [
                GoogleSearch(),
                Bash(),
                StrReplaceEditor()
            ]
        
        # Add tools to agents
        for tool in tools:
            execution_agent.add_tool(tool)
            orchestrator.register_tool(tool)
        
        # Register agents
        orchestrator.register_agent(planning_agent)
        orchestrator.register_agent(execution_agent)
        
        # Create nodes
        planner_node = orchestrator.create_agent_node(planning_agent.name, "plan")
        executor_node = orchestrator.create_agent_node(execution_agent.name, "execution_result")
        
        # Create a sequential workflow
        workflow = orchestrator.create_sequential_workflow(
            nodes=[planner_node, executor_node],
            flow_name="PlanningWorkflow"
        )
        
        # Run the workflow
        result = orchestrator.run_workflow(
            workflow_name="PlanningWorkflow",
            inputs={"task": task}
        )
        
        return result
    
    @staticmethod
    async def run_agent_rag_workflow(
        query: str,
        documents: List[Dict[str, Any]],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a RAG (Retrieval-Augmented Generation) workflow.
        
        This example creates a workflow that retrieves relevant documents based on a query
        and then generates a response using an agent.
        
        Args:
            query: The query to search for
            documents: List of documents to search through
            context: Additional context for the task
            
        Returns:
            The results from the workflow execution
        """
        # Initialize the orchestrator
        orchestrator = WorkflowOrchestrator()
        
        # Create a custom retrieval node
        retrieval_node = PocketFlowNode(name="retrieval")
        
        def retrieval_prep(shared):
            """Prepare data for retrieval."""
            return {
                "query": shared.get("query", ""),
                "documents": shared.get("documents", [])
            }
        
        def retrieval_exec(prep_res):
            """Execute retrieval logic."""
            query = prep_res.get("query", "")
            documents = prep_res.get("documents", [])
            logger.info(f"Retrieving documents for query: {query}")
            
            # Simple retrieval logic - in a real system, this would use embeddings
            relevant_docs = []
            for doc in documents:
                content = doc.get("content", "").lower()
                title = doc.get("title", "").lower()
                if query.lower() in content or query.lower() in title:
                    relevant_docs.append(doc)
            
            # Store the retrieval result
            retrieval_node.result = {
                "relevant_documents": relevant_docs,
                "count": len(relevant_docs)
            }
            
            return "default"
        
        def retrieval_post(shared, prep_res, exec_res):
            """Process retrieval results."""
            shared["retrieval_results"] = retrieval_node.result
            return shared
        
        # Attach methods to the retrieval node
        retrieval_node.prep = retrieval_prep
        retrieval_node.exec = retrieval_exec
        retrieval_node.post = retrieval_post
        
        # Create an agent for generating responses
        generator_agent = SimpleReActAgent(name="generator")
        orchestrator.register_agent(generator_agent)
        
        # Create nodes
        generator_node = orchestrator.create_agent_node(generator_agent.name, "response")
        
        # Create a sequential workflow
        workflow = orchestrator.create_sequential_workflow(
            nodes=[retrieval_node, generator_node],
            flow_name="RAGWorkflow"
        )
        
        # Run the workflow
        result = orchestrator.run_workflow(
            workflow_name="RAGWorkflow",
            inputs={
                "query": query,
                "documents": documents,
                "context": context or {}
            }
        )
        
        return result


async def main():
    """Main entry point for the workflow runner."""
    parser = argparse.ArgumentParser(description="Run integrated PocketFlow-OpenManus workflows")
    parser.add_argument(
        "--workflow", 
        type=str, 
        default="multi_agent",
        choices=["multi_agent", "planning_parallel", "rag"],
        help="Type of workflow to run"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="Create a simple Python script that fetches weather data from an API",
        help="Task to complete"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Query for RAG workflow"
    )
    parser.add_argument(
        "--documents", 
        type=str, 
        help="Path to JSON file containing documents for RAG workflow"
    )
    parser.add_argument(
        "--context", 
        type=str, 
        help="Path to JSON file containing additional context"
    )
    
    args = parser.parse_args()
    
    # Load context if provided
    context = None
    if args.context:
        try:
            with open(args.context, 'r') as f:
                context = json.load(f)
        except Exception as e:
            logger.error(f"Error loading context file: {e}")
            context = {}
    
    # Run the specified workflow
    if args.workflow == "multi_agent":
        logger.info(f"Running multi-agent workflow with task: {args.task}")
        result = await ExampleWorkflows.run_multi_agent_workflow(
            task=args.task,
            context=context
        )
        print("\nWorkflow result:")
        print(json.dumps(result, indent=2))
    
    elif args.workflow == "planning_parallel":
        logger.info(f"Running planning workflow with task: {args.task}")
        result = await ExampleWorkflows.run_planning_with_parallel_execution(
            task=args.task
        )
        print("\nWorkflow result:")
        print(json.dumps(result, indent=2))
    
    elif args.workflow == "rag":
        if not args.query:
            logger.error("Error: RAG workflow requires a query (--query)")
            return
            
        if not args.documents:
            logger.error("Error: RAG workflow requires documents (--documents)")
            return
            
        # Load documents
        try:
            with open(args.documents, 'r') as f:
                documents = json.load(f)
        except Exception as e:
            logger.error(f"Error loading documents file: {e}")
            return
            
        logger.info(f"Running RAG workflow with query: {args.query}")
        result = await ExampleWorkflows.run_agent_rag_workflow(
            query=args.query,
            documents=documents,
            context=context
        )
        print("\nWorkflow result:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())