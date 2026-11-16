"""Small helpers: an attribute-access dict and a batch-dimension helper."""


class AttrDict(dict):
    """Dict whose keys can also be read and set as attributes."""

    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        """Return the value stored under the key attr."""
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self):
        """Return the dict itself as the pickled state."""
        return self

    # Changed from upstream 7f48937: upstream assigned the state to the local name self here,
    # which has no effect, so the body is now pass.
    def __setstate__(self, state):
        """Do nothing: pickle restores the dict items itself."""
        pass


def add_batch_dim(batch):
    """Add a leading batch axis to an array, or to every array in a dict (in place)."""
    if isinstance(batch, dict):
        for key, value in batch.items():
            batch[key] = value[None]
    else:
        batch = batch[None]
    return batch
