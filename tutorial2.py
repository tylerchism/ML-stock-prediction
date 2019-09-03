game = [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]]

def game_board(game_map, player=0, row=0, column=0, just_display=False):
    print("   a  b  c")
    if not just_display:
        game[row][column] = player
    for count, element in enumerate(game):
        print(count, element)
    return game_map

def win(current_game):
    for row in game:
        print(row)
        if row.count(row[0]) == len(row) && row[0] != 0:
            print("Winner!")

for col in range(len(game)):
    check = []

    for row in game:
        check.append(row[col])

    if check.count(check[0]) == len(check) and check[0] != 0:
        print("Winner!")