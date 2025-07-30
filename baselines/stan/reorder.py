import pandas as pd

data = "caida"
syn = f'../../result/stan/{data}/syn.csv'
reordered = f'../../result/stan/{data}/reordered_{data}.csv'

def ip2int(ip):
    # ip should always be ipv4
    if isinstance(ip, str) and "." in ip and len(ip.split(".")) == 4:
        return sum([256**j*int(i) for j,i in enumerate(ip.split(".")[::-1])])
    elif isinstance(ip, str):
        return int(ip)
    elif isinstance(ip, int):
        return ip
    else:
        return 0
        try:
            return int(ip)
        except ValueError:
            raise ValueError("ip should be either string or int")


df = pd.read_csv(syn)
df["srcip"] = df["srcip"].apply(ip2int)
df["dstip"] = df["dstip"].apply(ip2int)
desired_order = ['srcip', 'dstip', 'srcport', 'dstport', 'proto', 'time', 'pkt_len', 'tos', 'flag', 'off', 'ttl']
df['off'] = 0  
df = df[['srcip', 'dstip', 'srcport', 'dstport', 'proto', 'time', 'pkt_len', 'tos', 'flag', 'off', 'ttl']]  # Reorder columns

df.to_csv(reordered, index=False)



