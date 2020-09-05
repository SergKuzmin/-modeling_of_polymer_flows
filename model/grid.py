from typing import Dict
import numpy as np


class GridKeyError(Exception):
    def __init__(self, key: str):
        self.key = key

    def __str__(self):
        if self.key == "(n+l)":
            return "key (n+l) not for set item"
        return "key must be 'n' or 'n+l' or 'n+l+1'"


class Grid:
    def __init__(self):
        self._dict: Dict[(str, np.array)] = {'n': None,
                                             'n+l': None,
                                             'n+l+1': None,
                                             '(n+l)': None}
        self._need_update_n_plus_l_in_brackets: bool = True

    def __setitem__(self, key, value):
        if key in self._dict:
            if key in {'n+l', 'n'}:
                self._dict[key] = value
                self._need_update_n_plus_l_in_brackets = True
                return
        raise GridKeyError(key)

    def __getitem__(self, item):
        if item in {'n+l', 'n'}:
            return self._dict[item]
        elif item == '(n+l)':
            if not self._need_update_n_plus_l_in_brackets:
                return self._dict[item]
            else:
                self._need_update_n_plus_l_in_brackets = True
                self._dict[item] = (self._dict['n+l'] + self._dict['n']) / 2
                return self._dict[item]
        elif item == 'n+l+1':
            return self.get_n_plus_l_plus_one()

    def get_n_plus_l_plus_one(self, ):
        pass


if __name__ == "__main__":
    a = Grid()
    a['n+l'] = 1
    a['n'] = 9
    print(a['(n+l)'])

