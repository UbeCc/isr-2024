import torch
import torch.nn as nn
import torch.optim as optim

class ActorEncoder(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_rma_obs,
                        num_actions,
                        num_latents,
                        actor_hidden_dims=[256, 256, 256],
                        enc_hidden_dims=[256, 256, 256],
                        activation=nn.ELU(),
                        **kwargs):
        super().__init__()

        # TODO ---------------------------------------------------
        # Please create actor and encoder
        # Both actor and encoder are MLPs with the given activation
        # Hidden dims of MLP layers are defined as actor_hidden_dims and enc_hidden_dims
        # You don't need to apply activation after the last layer.
        # Refer to algo/ppo/actor_critic.py for examples.
        # Take care of the input & output dims.
        # Encoder will get input of the rma obs (dim=num_rma_obs), and output a latent vector (dim=num_latents).
        # Actor will get input of actor obs (dim=num_actor_obs) as well as the latent vector, and output an action (dim=num_actions).

        mlp_input_dim_a = num_actor_obs + num_latents
        mlp_input_dim_e = num_rma_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Encoder
        enc_layers = []
        enc_layers.append(nn.Linear(mlp_input_dim_e, enc_hidden_dims[0]))
        enc_layers.append(activation)
        for l in range(len(enc_hidden_dims)):
            if l == len(enc_hidden_dims) - 1:
                enc_layers.append(nn.Linear(enc_hidden_dims[l], num_latents))
            else:
                enc_layers.append(nn.Linear(enc_hidden_dims[l], enc_hidden_dims[l + 1]))
                enc_layers.append(activation)
        self.encoder = nn.Sequential(*enc_layers)
        
        print(f"Actor MLP: {self.actor}")
        print(f"Encoder MLP: {self.encoder}")
        # --------------------------------------------------------

    def forward(self, observation, rma_obs):
        # TODO ---------------------------------------------------
        # Use your actor and encoder to get the action
        emb = self.encode(rma_obs)
        action_mean = self.actor(torch.cat([observation, emb], dim=-1))
        # --------------------------------------------------------
        return action_mean
    
    def encode(self, rma_obs):
        # TODO ---------------------------------------------------
        # Use your encoder to get the latent vector
        est_latent = self.encoder(rma_obs)
        # --------------------------------------------------------
        return est_latent