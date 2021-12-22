import argparse
import os


class ArgumentParserFileExtensionValidation(argparse.FileType):
    parser = argparse.ArgumentParser()
    def __init__(self, valid_extensions, file_name):
        self.valid_extensions = valid_extensions
        self.file_name = file_name

    def validate_file_extension(self):
        given_extension = os.path.splitext(self.file_name)[1][1:]
        if given_extension not in self.valid_extensions:
            self.parser.error(f"Invalid file format. Please provide a {self.valid_extensions} file")
        return self.file_name