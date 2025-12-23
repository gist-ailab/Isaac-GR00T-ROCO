from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

STATE_DIM = 16
ACTION_DIM = 16

custom_config = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "observations.images.head",
                "observations.images.left_hand",
                "observations.images.right_hand",
            ],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                'state'
            ],
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                'action'
            ],
            action_configs=[                
                ActionConfig(
                    rep=ActionRepresentation.RELATIVE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                    state_key="state",
                ),
            ]
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }

register_modality_config(custom_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)