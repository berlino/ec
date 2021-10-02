

IO_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T23:07:48.432156/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"
IO_NL_BIGRAM_CHECKPOINT = "experimentOutputs/arc/2021-09-30T15:06:29.861706/arc_aic=1.0_arity=0_BO=True_CO=True_ES=1_ET=720_t_zero=1_HR=0.0_it=5_MF=10_noConsolidation=True_pc=10_RS=1000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=200_TRR=randomShuffle_K=2_topkNotMAP=False_UET=3600_DSL=False_FTM=True.pickle"

def load_experiment_output(path):

	with open(path, "rb") as handle:
		result = dill.load(handle)
	return