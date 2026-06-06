import numpy as np
from aer.tools._tools import *


def test_exec_file():
    """ Execute a Python script file """
    # Consider a Python script "script.py" saved in the "/home" repertory
    # that contains the command `print("Hello World!")`
    exec_file("hello_world.py", globals())

def test_save_as_csv():
    """ Save a data set in a csv file  """
    save_as_csv("list.csv", [1, 2, 3])
    save_as_csv("double_list.csv", [[1, 2], [3, 4]])
    save_as_csv("double_list_tr.csv", [[1, 2], [3, 4]], True)

def test_load_as_csv():
    """ Load the data contained in a CSV file """
    # Save a simple list in a CSV file and load it
    save_as_csv("list.csv", [1, 2, 3])
    list = load_as_csv("list.csv")
    # Save a doubled transposed list in a CSV file and load it
    save_as_csv("double_list_tr.csv", [[1, 2], [3, 4]], True)
    double_list = load_as_csv("double_list_tr.csv", True)

def test_save_ws():
    """ Save a workspace as a shelve """
    # Save global variables
    var = 3
    vals = [0, 1, 2]
    data = np.arange(10, dtype=float)
    save_ws("workspace.out", ('var', 'vals', 'data'), locals())
    # Save local variables
    def func():
        data = [i for i in range(10)]
        print(locals())
        save_ws("data_func.out", vars_dict=locals())
    func()

def test_load_ws():
    """ Restore a shelved workspace """
    var = 3
    vals = [0, 1, 2]
    save_ws("workspace.out", ('var', 'vals'), locals())
    load_ws("workspace.out", None)

def test_get_ndim():
    """ Get the depth (number of dimensions) of an array """
    print(get_ndim(''), get_ndim(['']), get_ndim(['', '', '']))
    print(get_ndim('a'), get_ndim(['a']), get_ndim(['a', 'b', 'c']))
    print(get_ndim(0), get_ndim([0]), get_ndim([0, 1, 2]))
    print(get_ndim([[0]]), get_ndim([[0], [1], [2]]))
    print(get_ndim(np.empty(3)), get_ndim(np.empty((1, 2, 3))))
    print(get_ndim([0, [1, 2]]), get_ndim([[1, 2], 0]))

def test_flat_list():
    """ Flatten a 2D list """
    print(flat_list([[1, 2], [3, 4]]))

def test_make_iter():
    """ Check if `data` is scalar, and wrap it into a tuple if so """
    print(make_iter(1))
    print(make_iter([1, 2]))
    print(make_iter((i for i in range(5))))

def test_check_ext():
    """ Check the file name extension """
    print(check_ext("fname", 'csv'))
    print(check_ext("fname", '.csv'))
    print(check_ext("fname.csv", 'csv'))
    print(check_ext("fname.csv", 'txt'))

def test_check_folder():
    """ Check the validity of folder name """
    check_folder("test_dir")    # Folder created if it does not exist
    check_folder("test_dir")    # Nothing done since the folder exists

def test_check_path():
    """ Check the path of a file (folders + extension) """
    print(check_path("file", 'png'))
    print(check_path("file.png"))
    print(check_path("file.png", 'jpg'))
    print(check_path("dir/subdir/file", 'png'))
    print(check_path("dir/subdir/file.png", 'jpg'))

def test_check_keys():
    """ Dictionary check function """
    ref_dict = {'k1': 1, 'k2': 2, 'k3': 3}
    new_dict = {'k1': 1., 'k2': 2.}   # 'k3' missing
    # Check the dict: 'k1' and 'k2' will be taken from `new_dict`,
    # and 'k3' from 'ref_dict' as it is missing in `new_dict`
    fin_dict = check_keys(new_dict, ref_dict)


# Launch test/example functions
test_exec_file()

test_save_as_csv()

test_load_as_csv()

test_save_ws()

test_load_ws()

test_get_ndim()

test_flat_list()

test_make_iter()

test_check_ext()

test_check_folder()

test_check_path()

test_check_keys()

