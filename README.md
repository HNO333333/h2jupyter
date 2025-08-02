## usage

1. create a ssh config
    1. create a ssh key at your local computer
    2. add the key to server's authorized_keys
    3. more details: https://code.visualstudio.com/docs/remote/troubleshooting#_quick-start-using-ssh-keys
2. create config file: `cp config.yaml.example config.yaml`
3. change `hostname` in config.yaml
4. run `uv run server.py`