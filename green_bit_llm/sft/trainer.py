import shutil
import os
from typing import Optional

from trl import SFTTrainer

from green_bit_llm.common.utils import STRATEGY_FILE_NAME
from green_bit_llm.common.utils import get_model_path


class GbaSFTTrainer(SFTTrainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Saves the model to the specified directory and also ensures that the
        'quant_strategy.json' file is copied over to the same directory.

        Args:
            output_dir (Optional[str]): The directory to which the model and the
                                        'quant_strategy.json' file should be saved.
                                        If None, the model will be saved to the default location.
            _internal_call (bool): A flag used to indicate whether this method was
                                   called internally by the library, which can affect
                                   certain behaviors (not used in this override).

        Raises:
            ValueError: If the expected GBA prefix is not found in the output directory path.
        """

        # Perform the original save model behavior of the superclass
        super().save_model(output_dir)

        # save "quant_strategy.json" file
        if self.is_world_process_zero() and output_dir is not None:
            # out_dir should be os.path.join(args.save_dir, args.model)
            gba_prefix = "GreenBitAI" + os.path.sep

            start_index = output_dir.find(gba_prefix)
            if start_index != -1:
                start_pos = start_index + len(gba_prefix) - 1
                end_pos = output_dir.find(os.path.sep, start_pos + 1)

                if end_pos == -1:
                    model_name = output_dir[start_index:]
                else:
                    model_name = output_dir[start_index:end_pos]
            else:
                raise ValueError(f"GBA prefix not found in the path {output_dir}.")

            model_from_path = get_model_path(model_name)
            quant_strategy_file = os.path.join(model_from_path, STRATEGY_FILE_NAME)
            destination_path = os.path.join(output_dir, STRATEGY_FILE_NAME)
            shutil.copy(quant_strategy_file, destination_path)