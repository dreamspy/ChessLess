import chess
import chess.uci
import chess.pgn
import sys
import os
import fnmatch

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

PGN_FILES_FOLDER='pgnfiles'
PGN_FILES_PATTERN='*.pgn'

OUTPUT_FOLDER='datasets'

def find_files(directory, pattern):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(filename)
    return files

pgnfiles = find_files(PGN_FILES_FOLDER, PGN_FILES_PATTERN)

# Set up the chess engine
evaltime = 10 # thinking time for engine to evaluate each position (in ms).
handler = chess.uci.InfoHandler()
engine = chess.uci.popen_engine('./stockfish-8-mac/Mac/stockfish-8-64')

engine.info_handlers.append(handler)

# Play to the board the static evalution applies to and output it and its evaluation.
def output_evaluation(board, pvo, evalo ):
    tmp_board = board.copy()

    # handle cases with no PV (checkmate / stalemate)
    pv = pvo[1] if 1 in pvo else []

    for move in pv:
        tmp_board.push(move)

    # Give score of 100 if white checkmates, -100 if black. Implicitly gives 0 for stalemate
    eval = evalo[1].cp/100.0 if evalo[1].mate == None else evalo[1].mate * 100

    return [replace_tags(tmp_board.fen()), " ".join(map(str,pv)),str(eval)]

# Have engine evaluate the board
def evaluate(board):
    engine.position(board)
    evaluation = engine.go(depth=1)
    return output_evaluation( board, handler.info["pv"], handler.info["score"] )

for pgnfilename in pgnfiles:
    print('reading file', pgnfilename, file=sys.stderr)
    print('==========================', file=sys.stderr)
    with open(f'{PGN_FILES_FOLDER}/{pgnfilename}', 'r') as pgnfile, open(f'{OUTPUT_FOLDER}/{pgnfilename}.data', 'w') as datafile:

        # Loop through games.
        game = chess.pgn.read_game(pgnfile)
        while game != None:
            board = game.board()
            print(game.headers["Date"] if "Date" in game.headers else "No Date", file=sys.stderr)
            print(game.headers["Time"] if "Time" in game.headers else "No Time", file=sys.stderr)
            evaluate(board)
            for move in game.main_line():
                board.push(move)
                line = evaluate(board)
                datafile.write(':'.join(line)+'\n')
            game = chess.pgn.read_game(pgnfile)
    print()
