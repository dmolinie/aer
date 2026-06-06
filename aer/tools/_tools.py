""" Miscellaneous toolkit

Authors: Dylan MOLINIE
Company: Université Paris-Est Créteil, France
Contact: dylan.molinie@gmx.fr
Date: March 2021
Last revised: May 2026

License: GPLv3
"""

__all__ = ['COLORS',
    'save_as_csv', 'load_as_csv', 'save_ws', 'load_ws',
    'exec_file', 'get_ndim', 'flat_list', 'make_iter',
    'check_ext', 'check_folder', 'check_path', 'check_keys']

import os
import csv
import shelve

COLORS = (('blue', '#0050FF'),          # 0
          ('orange', '#FF8C00'),        # 1
          ('green', '#008000'),         # 2
          ('red', '#F00000'),           # 3
          ('darkviolet', '#8A2BE2'),    # 4
          ('brown', '#A52A2A'),         # 5
          ('pink', '#FFC0CB'),          # 6
          ('gray', '#808080'),          # 7
          ('darkkhaki', '#BDB76B'),     # 8
          ('turquoise', '#00CED1'),     # 9
          ('gold', '#FFD700'),          # 10
          ('lawngreen', '#7CFC00'),     # 11
          ('violet', '#EE82EE'),        # 12
          ('dodgerblue', '#1E90FF'),    # 13
          ('khaki', '#F0E68C'),         # 14
          ('black', '#000000'))         # 15


##############################################################################
##                              Miscellaneous                               ##
##############################################################################

#------------------------   Execute Python Script   -------------------------#
def exec_file(filename, global_vars=None, local_vars=None, root='./'):
    """ Execute a Python script file

    The purpose of this function is essentially to execute a set of
    configuration instructions at the very beginning of a script.
    This can automatize the import of a package (e.g. `numpy`), set the
    display version of it (`np.set_printoptions(legacy='1.21')`), etc.
    However, remember that "explicit" is better than "implicit".

    Parameters
    ----------
    filename : string
        The name (with path) of the file to execute.
    [OPT] global_vars : dict
        The dictionary of the global variables, typically `globals()`.
            :Default: None
    [OPT] local_vars : dict
        The dictionary of the local variables, typically `locals()`;
        if only `global_vars` is provided, `local_vars` defaults to it.
            :Default: None
    [OPT] root: string
        The path to the file; if 'home' or '~/', retrieve the `HOME`
        repertory using the `$HOME` environment variable (Linux only).
            :Default: './' (current directory)

    Returns
    -------
    None: directly execute the Python script.

    Examples
    --------
    # Consider a Python script "script.py" saved in the "/home" repertory
    # that contains the command `print("Hello World!")`

    >>> exec_file("/home/script.py", globals())
    'Hello World!'

    >>> exec_file("script.py", globals(), root="home")  # Linux only
    'Hello World!'
    """
    # pylint: disable=exec-used
    if root.lower() in ('home', '~/'):
        root = os.environ['HOME'] + '/'
    try:
        with open(root+filename, encoding='utf-8') as file:
            exec(file.read(), global_vars, local_vars)
    except FileNotFoundError:
        pass
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                            Save & Load Files                             ##
##############################################################################

#--------------------   Save Set of Values in CSV File   --------------------#
def save_as_csv(pathname, data, transpose=False):
    """ Save a data set in a csv file

    If the pathname contains directories, create them if needed. Also,
    check that `pathname` ends with '.csv'; append it if necessary.

    Parameters
    ----------
    pathname : str
        Pathname of the file in which to save the data.
    data : list, tuple or np.ndarray
        Data to be written in the file.
    [OPT] transpose : bool
        If the data should be written row-wise (`False`) or column-wise
        (`True`); in practice, if True, the `data` array is virtually
        transposed before being row-wise written.
            :Default: False (row-wise)

    Returns
    -------
    None : directly write into the file.

    Examples
    --------
    >>> save_as_csv('list.csv', [1, 2, 3])
    >>> save_as_csv('double_list.csv', [[1, 2], [3, 4]])
    >>> save_as_csv('double_list_tr.csv', [[1, 2], [3, 4]], True)
    """

    # Check that the folders exist and the filename ends with '.csv'
    pathname = check_path(pathname, '.csv')

    with open(pathname, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if transpose:
            max_lgt = max(len(row) for row in data) # Max length of any item
            data_cp = data.copy()
            # Fill any rows with empty spaces so that any has `max_lgt` items
            for row in data_cp:
                for _ in range(len(row), max_lgt):
                    row.append('')
            # Write the transposed data in the CSV file
            for j in range(max_lgt):
                writer.writerow([row[j] for row in data_cp])
        else:
            # Check if iterable (e.g. list or generator generator)
            if get_ndim(data) == 1:
                for row in data:
                    writer.writerow([row])
            # Wrap the item into a list so that it is written in a lone row
            else:
                for row in data:
                    writer.writerow(row)
#----------------------------------------------------------------------------#

#-----------------------   Load Data From CSV File   ------------------------#
def load_as_csv(pathname, transpose=False):
    """ Load the data contained in a CSV file

    Return a list of as many items as rows in the file, each item being
    itself a list of as many items as cols in the file; if the CSV file
    contains only one column, flatten it into a 1D list. If `transpose`
    is True, reverse the rows with the columns.

    Parameters
    ----------
    pathname : str
        Pathname of the file from which to load data. It should end with
        '.csv'; append it if necessary.
    [OPT] transpose : bool
        If the data should be loaded row-wise (`False`) or column-wise
        (`True`); in practice, if True, the `data` array is virtually
        transposed after being row-wise loaded.
            :Default: False (row-wise)

    Returns
    -------
    rows : list of (list of) strings
        The data rows of the CSV file.

    Examples
    --------
    # Save a simple list in a CSV file and load it
    >>> save_as_csv("list.csv", [1, 2, 3])
    >>> list = load_as_csv("list.csv")

    # Save a doubled transposed list in a CSV file and load it
    >>> save_as_csv("double_list_tr.csv", [[1, 2], [3, 4]], True)
    >>> double_list = load_as_csv("double_list_tr.csv", True)
    """

    # Check that the filename ends with '.csv'
    pathname = check_ext(pathname, '.csv')

    with open(pathname, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        rows = list(row for row in reader)  # Read data

    # If 'transpose', reverse the rows & columns
    if transpose:
        rows_cp = []
        for i in range(len(rows[0])):
            rows_cp.append([row[i] for row in rows if row[i] != ''])
        return rows

    # If there is only one column in the CSV file, flatten the 2D list
    if sum(len(row) for row in rows) == len(rows):
        return flat_list(rows)
    return rows
#----------------------------------------------------------------------------#

#-----------------------   Save Workspace (shelve)   ------------------------#
def save_ws(pathname, vars_to_save=None, vars_dict=None):
    """ Save a workspace as a shelve

    Take a dictionary of variables and save them in a shelve file; if
    the pathname contains directories, create them if needed. Save only
    the variables specified in `vars_dict`, or all else. If `vars_dict`
    is None, save the global variables of the `main` module (that spe-
    cified in `vars_to_save`, or all if None).

    Parameters
    ----------
    pathname : str
        Pathname of the file in which to save the workspace; no specific
        extension is required, but '.out' may be used.
    [OPT] vars_to_save : list or tuple of str
        Set of the names of the variables to save. If not specified,
        the whole workspace is saved (except modules, functions and
        private attributes (beginning with '__')).
            :Default: None
    [OPT] vars_dict : dict
        The dictionary of variables to save; if not provided or None,
        save the global variables of the `main` module. In particular,
        use `locals()` to save the local variables of a function.
            :Default: None

    Returns
    -------
    None : directly write into the file.

    Examples
    --------
    >>> import numpy as np

    # Save global variables
    >>> var = 3
    >>> vals = [0, 1, 2]
    >>> data = np.arange(10, dtype=float)
    >>> save_ws("workspace.out", ('var', 'vals', 'data'))

    # Save local variables
    >>> def func():
    ...     data = [i for i in range(10)]
    ...     save_ws("data_func.out", vars_dict=locals())
    >>> func()
    """

    # Shelve file
    file = shelve.open(check_path(pathname), 'n')

    # If no dictionary of variables provided, retrieve the global variables
    # of the '__main__' module
    if vars_dict is None:
        vars_dict = vars(os.sys.modules['__main__'])

    # If no key specified
    if vars_to_save is None:
        for key in vars_dict:
            # Ignore callable (function, class, etc.) and private attributes
            if key[0:2] == '__' or callable(vars_dict[key]):
                continue
            # Try to save the item (TypeError raised if it is a module)
            try:
                file[key] = vars_dict[key]
            except TypeError:
                pass
    # If the keys are specified
    else:
        if isinstance(vars_to_save, str):
            vars_to_save = [vars_to_save]
        for key in vars_to_save:
            # Ignore callable (function, class, etc.) and private attributes
            if key[0:2] == '__' or callable(vars_dict[key]):
                print("Functions, classes and private attributes ignored (`{key}`)")
            # Test if the keys are valid
            try:
                file[key] = vars_dict[key]
            except KeyError:
                print(f"No variable `{key}`")
            except TypeError:
                print(f"Cannot save modules (`{key}`)")

    file.close()
#----------------------------------------------------------------------------#

#----------------------   Restore Workspace (shelve)   ----------------------#
def load_ws(fname, vars_to_load=None, module=None):
    """ Restore a shelved workspace

    Take a shelved file and restore the data it contains as global varia-
    bles in the module provided as argument. Restore only that specified
    in the `vars_to_load` array, or all if it is None.

    Parameters
    ----------
    fname : str
        Pathname of the file from which to load & restore the workspace;
        no specific extension is required, but '.out' may be used.
    [OPT] vars_to_load : list or tuple of str
        Set of the names of the variables to restore. If not specified,
        the whole workspace is restored. Ignore the modules, functions
        and private attributes (those beginning with '__').
            :Default: None
    [OPT] module : string
        The module's namespace in which restore the data as global vars.
        The module must be loaded prior. If not provided, erroneous name
        or the module is not loaded, default to the `main` module.
            :Default: None

    Returns
    -------
    None : directly affect the data to the module's global variables.

    Examples
    --------
    >>> import numpy as np

    >>> var = 3
    >>> vals = [0, 1, 2]
    >>> data = np.arange(10, dtype=float)

    >>> save_ws("workspace.out", ('var', 'vals', 'data'))
    >>> load_ws("workspace.out", module='__main__')
    """

    # Shelve file
    file = shelve.open(fname, 'r')

    # If the module to which restore the variables as global variables
    # is invalid (or not loaded), default to the current 'main' module
    try:
        vars_dict = vars(os.sys.modules[module])
    except (TypeError, KeyError):
        vars_dict = vars(os.sys.modules['__main__'])

    # If no key specified
    if vars_to_load is None:
        for key in file:
            vars_dict[key] = file[key]

    # If the keys are specified
    else:
        if isinstance(vars_to_load, str):
            vars_to_load = [vars_to_load]
        for key in vars_to_load:
            try:
                vars_dict[key] = file[key]
            except KeyError:
                print(f"No variable '{key}'")

    file.close()
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                             Formatting Tools                             ##
##############################################################################

#-----------------------------   Get Columns   ------------------------------#
def get_ndim(array):
    """ Get the depth (number of dimensions) of an array

    Operate similarly to the `np.ndim` function, but can be used even when
    the array's items are not the same length; however, all items should
    have the same dim. as only the first is checked at each "recursivity".

    N.B.: does not work with arrays of `0` items (e.g. `np.empty(())`),
        as the shape is saved in an attribute and lost thereafter.

    Parameters
    ----------
    array : any type
        The array like for which to find the depth (dimension).
        Can be a scalar, string, tuple, np.ndarray, etc.

    Returns
    -------
    dim : int
        The dimension (depth) of the (first item of the) array.

    Examples
    --------
    >>> get_ndim(''), get_ndim(['']), get_ndim(['', '', ''])
    (0, 1, 1)
    >>> get_ndim('a'), get_ndim(['a']), get_ndim(['a', 'b', 'c'])
    (0, 1, 1)
    >>> get_ndim(0), get_ndim([0]), get_ndim([0, 1, 2])
    (0, 1, 1)
    >>> get_ndim([[0]]), get_ndim([[0], [1], [2]])
    (2, 2)
    >>> get_ndim(np.empty(3)), get_ndim(np.empty((1, 2, 3)))
    (1, 3)
    # Only the first item is checked, thus they should all have the same dim
    >>> get_ndim([0, [1, 2]]), get_ndim([[1, 2], 0])
    (1, 2)
    """
    dim = 0                     # Array dimension
    # Recursively run through the array, dimension by dimension
    while not isinstance(array, str):
        # Check if the subsequent array has an another subdimension
        try:
            array[0]
        # Otherwise, exit the loop at the next iteration
        except (IndexError, TypeError):
            break
        # Else, increment the dim and set `array` to its next dim `array[0]`
        else:
            dim += 1            # Increment the dimension
            array = array[0]    # Next dimension (exists as `try` succeeded)
    return dim
#----------------------------------------------------------------------------#

#--------------------------   Flatten a 2D List   ---------------------------#
def flat_list(lst):
    """ Flatten a 2D list """
    lst_flat = []
    for i in lst:
        lst_flat.extend(i)
    return lst_flat
#----------------------------------------------------------------------------#

#--------------------------   Make Data Iterable   --------------------------#
def make_iter(data):
    """ Check if `data` is scalar, and wrap it into a tuple if so

    Take some data and check if it is a tuple, list or np.ndarray; if so,
    do nothing and return `data`; else, assume `data` is a scalar and
    wrap it into a tuple to make it iterable.

    Parameters
    ----------
    data : string, scalar, array_like or generator
        The data to check, and to wrap if it is a scalar.

    Returns
    -------
    iterable : array_like or tuple
        The data, made iterable.

    Examples
    --------
    >>> make_iter(1)                        # (1,)
    >>> make_iter([])                       # []
    >>> make_iter([1, 2])                   # [1, 2]
    >>> make_iter((i for i in range(5)))    # Generator kept unchanged
    """
    # Check if iterable (e.g. generator)
    if hasattr(data, '__next__'):
        return data
    # Wrap a string into a tuple to make it fully accessible at once
    if isinstance(data, str):
        return (data,)
    # Else, check if it has a accessible components
    try:
        data[:1]
    except (IndexError, TypeError):
        return (data,)
    return data
#----------------------------------------------------------------------------#

##############################################################################



##############################################################################
##                            Checking Functions                            ##
##############################################################################

#-----------------------   Check Filename Extension   -----------------------#
def check_ext(fname, ext):
    """ Check the file name extension

    Take a filename and check if its last characters match the provided
    extension string; if `ext` does not start by a period '.', add one.

    Parameters
    ----------
    fname : string
        The filename to check.
    ext : string
        The file extension to check.

    Returns
    -------
    fname : string
        The formatted filename.

    Examples
    --------
    >>> check_ext("fname", 'csv')
    'fname.csv'
    >>> check_ext("fname", '.csv')
    'fname.csv'
    >>> check_ext("fname.csv", 'csv')
    'fname.csv'
    >>> check_ext("fname.csv", 'txt')
    'fname.csv.txt'
    """
    # Check that fname has a valid extension
    if ext[0] != '.':
        ext = '.' + ext
    if fname[fname.rfind('.'):] != ext:
        fname += ext
    return fname
#----------------------------------------------------------------------------#

#------------------------   Check Folder Validity   -------------------------#
def check_folder(folder):
    r""" Check the validity of folder name

    Take the name of a folder, create it if it does not exist, and add
    the OS-specific delimiter at the end of it if needed; the delimiter
    correctness is ensured by the `os` module ('/', '\\', ':', etc.).

    Parameters
    ----------
    folder : str
        The root name of the folder to check.

    Returns
    -------
    folder : str
        The folder path.

    Examples
    --------
    >>> check_folder("test_dir")    # Folder created if it does not exist
    >>> check_folder("test_dir")    # Nothing done since the folder exists
    """
    # Create `folder` if it does not exist
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    # Check that `folder` ends with the correct delimiter (/, \, etc.)
    return os.path.join(folder, '')     # Ensure compat. between OSes
#----------------------------------------------------------------------------#

#----------------   Check File Path (Folders + Extension)   -----------------#
def check_path(pathname, ext=None):
    """ Check the path of a file (folders + extension)

    Take the path of a file and split it into its directories and name.
    If any folders of the path do not exist, create the missing ones in
    memory. Also, if an extension is provided as argument, check that
    the filename ends with this extension, or append it else. Finally,
    return the corrected path, with the folders created in-between.

    N.B.: if the purpose is only to check the file extension, and not
        that the folders exist, prefer using the `check_ext` function.

    Parameters
    ----------
    pathname : str
        Pathname of the file from which to load & restore the workspace;
        no specific extension is required, but '.out' may be used.
    [OPT] ext : str
        If provided, the extension of the file pointed by the path is
        checked and compared to `ext`; if they do not match, the `ext`
        extension is appended to the filename in the corrected path.
        If not provided, just ignore that checking.
            :Default: None (no extension checking)

    Returns
    -------
    pathname : str
        The corrected path of the file; the possibly missing folders of
        the path are created in-between.

    Examples
    --------
    >>> check_path("file", 'png')
    'file.png'
    >>> check_path("file.png")
    'file.png'
    >>> check_path("file.png", 'jpg')
    'file.png.jpg'
    >>> check_path("dir/subdir/file", 'png')
    'dir/subdir/file.png'
    >>> check_path("dir/subdir/file.png", 'jpg')
    'dir/subdir/file.png.jpg'
    """
    folder, fname = os.path.split(pathname)
    if folder != '' and not os.path.exists(folder):
        os.makedirs(folder)
    if ext is not None:
        fname = check_ext(fname, ext)
    return os.path.join(folder, fname)
#----------------------------------------------------------------------------#

#-------------------------   Dictionary Checking   --------------------------#
def check_keys(params, rparams):
    """ Dictionary check function

    Take a dictionary to test and compare it to a reference, key by key.
    For every of the reference, check if the dict to test has the same;
    if so, take the value of the dict to test, or keep the ref otherwise.
    If a key is missing, display a warning message (in argument).

    Parameters
    ----------
    params : dict
        The dictionary to test.
    rparams : dict
        The dictionary of reference, whose keys are used as keys for the
        dictionary to test (params). If params has a valid key, take its
        value; keep the reference otherwise.

    Returns
    -------
    kparams : dict
        The dictionary composed of the valid entries of the params dict,
        and of the references of rparams dict for the invalid entries.

    Examples
    --------
    >>> ref_dict = {'k1': 1, 'k2': 2, 'k3': 3}
    >>> new_dict = {'k1': 1., 'k2': 2.}   # 'k3' missing

    # Check the dict: 'k1' and 'k2' will be taken from `new_dict`,
    # and 'k3' from 'ref_dict' as it is missing in `new_dict`
    >>> fin_dict = check_keys(new_dict, ref_dict)
    """

    # Check if the input `params` is a dictionary
    if not isinstance(params, dict):
        return rparams

    # Check the params' keys
    kparams = {}
    for k in list(rparams.keys()) + list(params.keys()):
        if k in params.keys() and k in rparams.keys():
            if isinstance(params[k], dict) or isinstance(rparams[k], dict):
                kparams[k] = check_keys(params[k], rparams[k])
            else:
                kparams[k] = params[k]
        elif k in params.keys():
            kparams[k] = params[k]
        else:
            kparams[k] = rparams[k]

    return kparams
#----------------------------------------------------------------------------#

##############################################################################
