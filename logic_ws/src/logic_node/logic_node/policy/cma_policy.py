from policy.base_policy import BasePolicy
from model.cma_net import CMANet


class CMAPolicy(BasePolicy):
    
    def __init__(
        self,
        observation_space,
        action_space
    ) -> None:
        super().__init__(
            CMANet(
                observation_space=observation_space,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    def passfunc(self):
        pass

