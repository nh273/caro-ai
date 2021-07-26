## Setup
The environment is managed with Pipenv. `pipenv install` to install all dependencies from the Pipfile.
Then use `pipenv shell` to load the environment to your shell.
Alternatively you can do `pipenv run [command]` to run commands from within the Pipenv environment.

## Training
Train the model with `python train.py -n [any name you want for this run]`

## Playing
You can interact with the bot via Telegram.
On Telegram, talk to @botfather to create a new bot and obtain its API key.
Then create `.config/bot.ini` and add that API key as follow:
```
[telegram]
api=<API KEY HERE>
```
`telegram` should be in square brackets. Your API KEY can just be pasted without any brackets or quotes.
Run Telegram bot with `python telegram-bot.py -m [directory with model files] -l [path to log file]`