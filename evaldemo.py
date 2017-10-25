import chess
import chess.uci
import chess.pgn
import sys

def replace_tags(board):
    board_san = board.split(" ")[0]
    board_san = board_san.replace("2", "11")
    board_san = board_san.replace("3", "111")
    board_san = board_san.replace("4", "1111")
    board_san = board_san.replace("5", "11111")
    board_san = board_san.replace("6", "111111")
    board_san = board_san.replace("7", "1111111")
    board_san = board_san.replace("8", "11111111")
    for i in range(len(board.split(" "))):
        if i > 0 and board.split(" ")[i] != '':
            board_san += " " + board.split(" ")[i]
    return board_san

# Command line arguments
arguments = sys.argv
pgnfilename = str(arguments[1])

# Set up the chess engine
evaltime = 10 # thinking time for engine to evaluate each position (in ms).
handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('./stockfish-8-mac/Mac/stockfish-8-64')
#engine = chess.uci.popen_engine('./stockfish-8-64')
engine.info_handlers.append(handler)


# Play to the board the static evalution applies to and output it and its evaluation.
def output_evaluation( board, pvo, evalo ):
    tmp_board = board.copy();
    pv = pvo[1] if 1 in pvo else []
    for move in pv:
        #print(move)
        tmp_board.push(move)
#    print(tmp_board)
    eval = evalo[1].cp/100.0 if evalo[1].mate == None else evalo[1].mate * 100
    print(replace_tags(tmp_board.fen()), ":" ,eval, sep = "")
    #print(eval)

# Have engine evaluate the board
def evaluate(board):
    engine.position(board)
    #evaluation = engine.go(movetime=evaltime)
    evaluation = engine.go(depth=1)
    #print(evaluation)
    #print( board.fen() )
    #print( board )
    #print ('Board evaluation', handler.info["score"][1].cp/100.0)
    #print ('Best move', board.san(evaluation[0]))
    #print ('Principal variation: ', board.variation_san(handler.info["pv"][1]))
    #print(handler.info["pv"], file=sys.stderr)
    #print(handler.info["score"], file=sys.stderr)
    output_evaluation( board, handler.info["pv"], handler.info["score"] )

#Read pgn file:
f = open(pgnfilename)

# Loop through games.
game = chess.pgn.read_game(f)
while game != None:
#    print(game)
    board = game.board()
    print(game.headers["Date"] if "Date" in game.headers else "No Date", file=sys.stderr)
    print(game.headers["Time"] if "Time" in game.headers else "No Time", file=sys.stderr)
    evaluate(board)
    for move in game.main_line():
        #print(move)
        board.push(move)
        evaluate(board)
    game = chess.pgn.read_game(f)

f.close()
