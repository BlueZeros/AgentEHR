from .ABCAgent import ABCAgent
from .BaseAgent import BaseAgent
from .MCPAgent import MCPBaseAgent
from .MCPReflexionAgent import MCPReflexionAgent
from .MCPReSumAgent import MCPReSumAgent
from .MCPReflecToolAgent import MCPReflecToolAgent
from .MCPRetroSumAgent import MCPRetroSumAgent
from .MCPReasoningBankAgent import MCPReasoningBankAgent

from agentlite.logging import AgentLogger
from agentlite.commons import EHRManager
from agentlite.llm import get_llm_backend
from agentlite.agent_prompts.task_prompt import ROLE_PROMPT

AGENT_TYPE = {
    "react": BaseAgent,
    "mcp": MCPBaseAgent,
    "mcp_reflexion": MCPReflexionAgent,
    "mcp_resum": MCPReSumAgent,
    "mcp_reflectool": MCPReflecToolAgent,
    "mcp_reasoningbank": MCPReasoningBankAgent,
    "mcp_retrosum": MCPRetroSumAgent
}

def get_agent(data_args, model_args, agent_args):
    logger = AgentLogger(
        log_file_name=f"{data_args.output_path}/agent.log",
        PROMPT_DEBUG_FLAG=data_args.debug,
        SAVE_TO_JSON=True,
    )
    
    llm = get_llm_backend(model_args)

    if "mcp" not in agent_args.agent_type:
        ehr_manager = EHRManager(
            data_path=data_args.ehr_path
        )

        agent = AGENT_TYPE[agent_args.agent_type](
            name="EHR_agent",
            role=f"{ROLE_PROMPT}",
            llm=llm,
            ehr_manager=ehr_manager,
            logger=logger,
            **vars(agent_args)
        )
    else:
        agent = AGENT_TYPE[agent_args.agent_type](
            name="EHR_agent",
            role=f"{ROLE_PROMPT}",
            llm=llm,
            logger=logger,
            **vars(agent_args)
        )
    return agent