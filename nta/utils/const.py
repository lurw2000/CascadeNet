EPSILON = 1e-12
interarrival_eps = EPSILON

# an incomplete list of protocols
INT2PROTO = {
    1: "ICMP",       # Internet Control Message Protocol
    6: "TCP",        # Transmission Control Protocol
    17: "UDP",       # User Datagram Protocol
    2: "IGMP",       # Internet Group Management Protocol
    4: "IPv4",       # IPv4 encapsulation
    41: "IPv6",      # IPv6 encapsulation
    47: "GRE",       # Generic Routing Encapsulation
    50: "ESP",       # Encapsulating Security Payload
    51: "AH",        # Authentication Header
    89: "OSPF",      # Open Shortest Path First
    112: "VRRP",     # Virtual Router Redundancy Protocol
    115: "L2TP",     # Layer Two Tunneling Protocol
    88: "EIGRP",     # Enhanced Interior Gateway Routing Protocol
}

PROTO2INT = {v: k for k, v in INT2PROTO.items()}