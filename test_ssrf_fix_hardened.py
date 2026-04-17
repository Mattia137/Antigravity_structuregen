import socket
from urllib.parse import urlparse
import ipaddress

def is_safe_url(url):
    """
    Validates if the provided URL is safe to use for remote requests.
    Restricts protocols to http/https and prevents access to private IP ranges.
    Uses DNS resolution to verify the actual IP of the hostname.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Use DNS resolution to get the IP address of the hostname
        # This helps protect against DNS rebinding and attempts to hide private IPs behind domains.
        try:
            # We use getaddrinfo to handle both IPv4 and IPv6
            for res in socket.getaddrinfo(hostname, None):
                ip_str = res[4][0]
                ip = ipaddress.ip_address(ip_str)
                # If any of the resolved IPs are private/internal, reject the URL
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                    return False
        except (socket.gaierror, ValueError):
            # Could not resolve hostname or invalid IP format
            return False

        return True
    except Exception:
        return False

def test_is_safe_url():
    test_cases = [
        ("http://localhost:5000", False), # Localhost now blocked
        ("https://127.0.0.1:5000", False), # Loopback now blocked
        ("http://0.0.0.0:5000", False), # Any-interface now blocked
        ("http://google.com", True),
        ("https://google.com", True),
        ("ftp://google.com", False),
        ("http://192.168.1.1", False),
        ("http://10.0.0.1", False),
        ("http://172.16.0.1", False),
        ("http://169.254.169.254", False),
        ("not_a_url", False),
        ("", False),
    ]

    for url, expected in test_cases:
        result = is_safe_url(url)
        print(f"URL: {url}, Expected: {expected}, Result: {result}")
        assert result == expected

if __name__ == "__main__":
    test_is_safe_url()
    print("All is_safe_url tests passed!")
