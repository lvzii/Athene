import sys
from enum import Enum, unique
from .train.train import run_exp


@unique  # unique起什么作用
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    ENV = "env"
    EVAL = "eval"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VER = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1)

    if command == Command.TRAIN:
        pass
