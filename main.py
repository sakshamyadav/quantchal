import qe_minesweeper

magnetic_field, dipole = qe_minesweeper.load_dataset("C:\\Users\\saksh\\Desktop\\dataset\\stage1_training_dataset.h5", 1)

mine_locations = qe_minesweeper.load_answers("C:\\Users\\saksh\\Desktop\\dataset\\stage1_training_dataset.h5", 666)
mag_east = magnetic_field[0]
mag_north = magnetic_field[1]
mag_up = magnetic_field[2]

mine_loc_east = mine_locations[0]
mine_loc_north = mine_locations[1] 
mine_1 = (mine_loc_east[0], mine_loc_north[0])
print(mag_east)