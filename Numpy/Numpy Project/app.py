import streamlit as st
import numpy as np

if "board" not in st.session_state:
    st.session_state.board = np.zeros((3, 3), dtype=int)

if "player" not in st.session_state:
    st.session_state.player = 1

board = st.session_state.board

def check_winner(board):
    for row in board:
        if np.sum(row) == 3:
            return "X"
        if np.sum(row) == -3:
            return "O"

    for col in range(3):
        if np.sum(board[:, col]) == 3:
            return "X"
        if np.sum(board[:, col]) == -3:
            return "O"

    if np.sum(np.diag(board)) == 3:
        return "X"
    if np.sum(np.diag(board)) == -3:
        return "O"

    if np.sum(np.diag(np.fliplr(board))) == 3:
        return "X"
    if np.sum(np.diag(np.fliplr(board))) == -3:
        return "O"

    return None

def make_move(r, c):
    if board[r][c] == 0:
        board[r][c] = st.session_state.player
        st.session_state.player *= -1

st.title("🎮 Tic Tac Toe (Numpy + Streamlit)")

symbols = {0: " ", 1: "X", -1: "O"}

for r in range(3):
    cols = st.columns(3)

    for c in range(3):
        cols[c].button(
            symbols[board[r][c]],
            key=f"{r}-{c}",
            on_click=make_move,
            args=(r, c)
        )

winner = check_winner(board)

if winner:
    st.success(f"🏆 Player {winner} Wins!")

elif not np.any(board == 0):
    st.info("🤝 Draw!")

if st.button("Restart Game"):
    st.session_state.board = np.zeros((3, 3), dtype=int)
    st.session_state.player = 1
    st.rerun()