from setuptools import setup


def main():
    setup(console_scripts=["llamafactory-cli = llamafactory.cli:main"])


if __name__ == "__main__":
    main()
