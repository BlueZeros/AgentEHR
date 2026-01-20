from typing import Optional
# from pydantic import BaseModel


class AgentAct:
    """Using AgentAct class to design the agent self-actions and API-call actions

    :param name: action name
    :type name: str
    :param desc: the description/documents of this action
    :type desc: str, optional
    """
    def __init__(
        self,
        name: str,
        desc: str = None,
        params: Optional[dict] = None
    ):

        self.name = name
        self.desc = desc
        self.params = params


ActObsChainType = list[tuple[AgentAct, str]]
