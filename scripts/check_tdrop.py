import torch

model_dir = "/gscratch/ark/ivyg/fasttext-debias/models/mbart-roen"
state_dict = torch.load(model_dir + "/model.pt")

print(state_dict.keys())

# enro
# print(state_dict["args"].dropout)

# else
print(state_dict["cfg"]["model"].dropout)
print(state_dict["cfg"]["task"].dropout)

dropout = 0.1

# enro
# state_dict["args"].dropout = dropout
# print(state_dict["args"].dropout)

# else
state_dict["cfg"]["model"].dropout = dropout
state_dict["cfg"]["task"].dropout = dropout
print(state_dict["cfg"]["model"].dropout)
print(state_dict["cfg"]["task"].dropout)

torch.save(state_dict, model_dir + f"/model.tdrop{dropout}.pt")
