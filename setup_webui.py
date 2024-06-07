import subprocess
import sys

def install_node():
    try:
        subprocess.check_call(['node', '--version'])
    except subprocess.CalledProcessError:
        print("Node.js is not installed. Please install Node.js from https://nodejs.org/")

def install_dependencies():
    print("Installing NPM dependencies...")
    subprocess.check_call(['npm', 'install'], cwd='webui')

def build_project():
    print("Building the project...")
    subprocess.check_call(['npm', 'run', 'build'], cwd='webui')

if __name__ == "__main__":
    install_node()
    install_dependencies()
    build_project()
    print("Web UI has been successfully installed and built.")