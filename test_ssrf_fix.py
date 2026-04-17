from urllib.parse import urlparse
import ipaddress

def is_safe_url(url):
    """
    Validates if the provided URL is safe to use for remote requests.
    Restricts protocols to http/https and prevents access to private IP ranges.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        # Explicitly allow localhost for development
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            return True
        # Check if the hostname is a private IP
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast:
                return False
        except ValueError:
            # Not an IP address, could be a domain.
            pass
        return True
    except Exception:
        return False

def test_is_safe_url():
    test_cases = [
        ("http://localhost:5000", True),
        ("https://localhost:5000", True),
        ("http://127.0.0.1:5000", True),
        ("http://0.0.0.0:5000", True),
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
