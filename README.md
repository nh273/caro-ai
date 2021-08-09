Implementing the [AlphaZero architecture](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go) for the game Connect4 with guidance from the book [Deep Reinforcement Learning Hands-on](https://www.amazon.com/Deep-Reinforcement-Learning-Hands-optimization-ebook-dp-B07ZKDLZCR/dp/B07ZKDLZCR/ref=mt_other?_encoding=UTF8&me=&qid=).
Majorly refactored the codebase to be easily extensible with new games. Added [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game) class and trained model to play Tic-Tac-Toe.
# Practical Stuffs
## Setup
The environment is managed with [Pipenv](https://pipenv.pypa.io/en/latest/install/). From your project directory, run `pipenv install` to install all dependencies from the Pipfile.
Then each time you want to load the virtual environment to your shell, simply run `pipenv shell` from the project directory.
Alternatively you can use `pipenv run [command]` to run commands from within the Pipenv environment.
If you ever need to remove the virtual environment, you can do so with `pipenv --rm`.
## Choose game
Each of the commands below can be run with a `-g` flag to specify the game you are training or playing with. `-g 0` for Connect4 and `-g 1` for TicTacToe.
## Training
Train the model with `python train.py -g [game] -n [any name you want for this run]`. Trained model will be saved to `saves/[run name]/[auto-generated-model-name].dat`.
Training statistics can be observed using TensorBoard. Start a session with `tensorboard --logdir .` and TensorBoard will open in browser (by default at
[http://localhost:6006/](http://localhost:6006/)). From TensorBoard you can observe the network losses at each training steps, and the winning ratio of the challenger against the current best model. The latter graph will likely be increasing up to a point before sharply dropping as the best model is replaced by a successful-enough challenger, before increasing again as yet a new challenger gets better.
2 trained models for each implemented game have been included for your playing pleasure at `saves/trained_tictactoe` and `saves/trained_connect4`.
## Human Playing Against a Trained Model
You can interact with the bot via [Telegram](https://desktop.telegram.org/).
On Telegram, talk to @botfather to create a new bot and obtain its API key.
Then create `.config/bot.ini` and add that API key as follow:
```
[telegram]
api=<API KEY HERE>
```
`telegram` should be in square brackets. Your API KEY can just be pasted without any brackets or quotes.
Run Telegram bot with `python telegram-bot.py -g [game] -m [directory with model files]`. E.g:
 `python telegram-bot.py -g 0 -m saves/trained_connect4/`
Create a chat in Telegram with the bot you created. Send the command `\list` to see all the models available inside your model directory, `\play [model index]` to start a game against a model in the list.
## Let Models Play Against Each Other
Models can also play against one another using
`python play.py -g [game] model1_filename model2_filename ...` e.g.:
`python play.py -g 0 saves/trained_connect4/best_025_10600.dat saves/test/best_026_12000.dat`
`python play.py -g 1 saves/trained_tictactoe/best_005_00900.dat saves/test/best_004_00800.dat`
You can also specify the number of rounds that the models will play with the argument `-r`
The script will print a leaderboard of models with wins, losses, and draws
# More Details
## About AlphaZero
AlphaZero is a reinforcement learning architecture that can learn to play [perfect information](https://en.wikipedia.org/wiki/Perfect_information) 2-player board games without human domain knowledge or training data. This is [a very good technical explanation(https://web.stanford.edu/~surag/posts/alphazero.html)] of AlphaZero, but in summary:
1. A neural network with random weights is initialized, it make random moves.
2. The network plays against itself a number of matches to completion. Now we have some data on the value of each action at each game state. This generates training data.
3. The network is trained against data generated in step 2. It (maybe) gets better at making moves. This is assessed by letting the newly trained network play against the old network.
4. If the new network beats the old network (by a certain proportion), then it becomes the new "best" network that will self-play to generate training data to create a better network, and so on.

More details to follow in each component.
## Components
### MCTS
`lib/mcts.py`
This is arguably the core of AlphaZero. You can think of MCTS as AlphaZero "thinking ahead" from a given game state. MCTS is essentially conducting a game tree, i.e. a graph of game states with edges being actions that lead from one state to another. For every game state `s` that have been explored, and for each action `a` at each game state `s`, the following data are tracked:
1. Number of times the action `a` had been taken at game state `s`: `N(s,a)`
2. [Expected reward](https://en.wikipedia.org/wiki/Q-learning) of action `a`: `Q(s,a)`
3. A prior probability of taking action `a`, predicted by the (currently best) neural network, given game state `s`: `p(s,a)`
From the above (and a hyperparameter `c_puct`) we can calculate an adjusted upper confidence bound for the Q-values of each action at each game state `U(s,a)`.
For most games there are obviously more game states than it is feasible to explore, but the above value can serve as an heuristic to choose which game states to examine.
In `lib/mcts.py`, the values above are stored, each in one `dict` with the keys being the integer form of the game state, each state accessing a list of values corresponding to the game actions. Theoretically the game state can be any indexable type, but for this implementation the `int` type is chosen as the neural net accepts arrays of numbers as inputs and it is fairly straightforward in most cases to convert the game state from a single integer to a form that can be accepted by the neural net.
#### State search
For each turn of the simulation: given a state, choose the action that has highest U. Pass it to the game logic, which returns a new game state. If the new state is found, look up U in the dict. Else, add the state to a queue for node expansion later (this node expansion is done in batches to be more efficient with querying values from the Pytorch neural net)
#### Node Expansion and Backup
With a queue of new, unencountered states, if the state is not terminal (win, lose, or draw) query the neural net for its prediction of probabilities of each action at each game state and the overall predicted game value at that state. Create new nodes in the MCTS tree, i.e. adding the new state to each dictionary with predicted probabilities, 0's for action counts and values.
If the state is terminal, we get the actual value: -1 for lost, 0 for draw, +1 for win.
We also perform backup: update the game value and visit count along the path taken so far.
#### Batch search and mini-batch search
The bottleneck of the MCTS process is querying the neural net to expand new tree nodes. To be more efficient with this, we query the neural net in batches of several leaf states (`search_minibatch()`). However, this is not optimal in the early stages of MCTS where the game tree is not very populated. As we only back up the values and node counts after a batch of querying, the MCTS ends up repeating itself several times in a minibatch. Thus to expand the tree more with each MCTS step, we perform several of these minibatch searches (`search_batch()`).
#### Get policy value
For the MCTS tree search process, we deterministically select the action with the highest value at each game state. But for actual playing (including self-play to generate training data), we stochastically select an action from the state tree based on how often it has been chosen, as the MCTS process makes good actions more often selected. The degree of exploration is controlled by a hyperparameter Tau. In the AlphaZero paper, for the first 30 moves Tau is set to 1 (max exploration), so the actual move is a weighted random choice with the probabilities being the normalized visit count of each action. After 30 moves, Tau = 0, i.e. the model chooses the most visited move always. The number of steps before setting Tau = 0 is a tunable hyperparameter (`config.py`). It should be set to a smaller value for simpler games
#### Edge Cases & Gotchas
For the root node (inital game state) where there are no data to calculate yet, we generate probabilities using a [Dirichlet distribution](https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper).
The value we received should always be reversed before being added to the tree nodes. After making a move, the "point-of-view" of the game is reversed (it's the other player's turn, if they won, you've lost).
### Model
`lib/model.py`
It's a neural net with 4 convolutional layers with batch normalization, using Leaky ReLU for activation, as suggested by the book. The network accepts 2D arrays of arbitrary size with 2 channels (one for each player). I have not tested a deeper network. The network outputs a policy head: array equal in length to the action space, which we then softmax to get the probabilities, and a value head: single estimate of the reward at that game state.
### Game
The logic of each game. Check `lib/game/game.py` for the expected interface. The game needs to be able to represent the game state as an integer for the MCTS. The simplest way of doing this is probably the implementation of TicTacToe: use one digit for each player's token, and one for empty squares. Serialize the game board into a sequence of single digits.
The game also needs to update the game state after each move (which is also expected as a single integer), determine if the move led to a final outcome, get a list of legal and illegal moves given the game state.
Finally, the game is responsible for transforming its game state into a list of inputs to train the neural network. Following the AlphaZero paper, the input is a 2-channel 2-D array, with each channel being the position of one player's tokens on the game board. The MCTS will batch game states together in a list to train the network in batches, so the game should be able to convert a list of game states to a list of network inputable arrays.
To add new games, simply add another module in the `lib/game` folder and implement the `BaseGame`interface defined in `lib/game/game.py`. A catalogue of available games is kept in `lib/game/game_provider.py`. Modify this to provide your game for the train, play and telegram-bot scripts.
## Hyperparameters
All hyperparameters can be found in `config.py`. Values are from the AlphaZero paper for Go, unless otherwise stated, except for MCTS_SEARCHES and MCTS_BATCH_SIZE, which are implementation quirks for PyTorch from the book Deep Reinforcement Learning Hands-on. I have not experimented with tuning these hyperparameters, but looking to do so in the near future.