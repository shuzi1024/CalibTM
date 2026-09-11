from exp.exp_imputation import Exp_Imputation as BaseExp


class Exp_Imputation(BaseExp):
    def prepare_batch(self, batch_x):
        batch_x = super().prepare_batch(batch_x)
        if batch_x.shape[-1] < self.args.c_out:
            raise ValueError(
                f"GEANT batch has {batch_x.shape[-1]} features, but c_out={self.args.c_out}."
            )
        return batch_x[:, :, :self.args.c_out]
