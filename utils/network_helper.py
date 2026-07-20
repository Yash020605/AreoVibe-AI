import socket

def get_local_ip():
    """
    Attempts to get the local IPv4 address of the machine.
    This works by creating a dummy UDP connection that doesn't 
    actually send packets but asks the OS for the easiest route out.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Attempt to connect to an external IP
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        # Fallback to localhost if network is unreachable
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP
