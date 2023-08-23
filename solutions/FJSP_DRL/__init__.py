from gym.envs.registration import register

# Registrar for the gym environment
# https://www.gymlibrary.ml/content/environment_creation/ for reference
register(
    id='fjsp-v0',  # Environment name (including version number)
    entry_point='solutions.FJSP_DRL.env:FJSPEnv',  # The location of the environment class, like 'foldername.filename:classname'
)