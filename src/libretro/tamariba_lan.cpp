// Tamariba: the access point's internet, through the Tamariba Plaza (tamariba_lan.h).

#include "tamariba_lan.h"
#include "tamariba_net.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

namespace TamaribaLAN
{

namespace
{

const u32 kSubnet = 0x0A400000; // 10.64.0.0/24
const u32 kRouterIP = kSubnet | 0x01;
const u32 kDNSIP = kSubnet | 0x02;
const u32 kClientIP = kSubnet | 0x10;
const u8 kRouterMAC[6] = {0x00, 0xAB, 0x33, 0x28, 0x99, 0x44};

const size_t kMaxQueuedFrames = 512;
const size_t kMaxTcp = 32;
const size_t kMaxUdp = 32;
const size_t kMaxDns = 16;
const u32 kOurMSS = 1360;           // fits the console's 1500-byte MTU with room to spare
const size_t kBufferToInet = 16384; // what we take from the console before the exit takes it
const size_t kBufferToDS = 65536;   // what we read from the exit before the console acks it
const u64 kConnectTimeoutUs = 20000000;
const u64 kDnsTimeoutUs = 8000000;
const u64 kIdleTimeoutUs = 180000000;
const u64 kRetransmitUs = 400000;

const tamariba_inet_interface* Inet = nullptr;
bool Inited = false;

u8 ClientMAC[6];
bool HaveClientMAC = false;
u16 IPv4ID = 0;
u32 NextISS = 0x1B2C3000;

std::deque<std::vector<u8>> ToDS;
Stats Counters = {};

u64 NowUs()
{
    return (u64)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

u16 Be16(const u8* p) { return (u16)((p[0] << 8) | p[1]); }
u32 Be32(const u8* p) { return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | p[3]; }
void PutBe16(u8* p, u16 v) { p[0] = (u8)(v >> 8); p[1] = (u8)v; }
void PutBe32(u8* p, u32 v) { p[0] = (u8)(v >> 24); p[1] = (u8)(v >> 16); p[2] = (u8)(v >> 8); p[3] = (u8)v; }

// Network byte order (the interface's) <-> host-order numbers (ours).
u32 FromNet(u32 ipv4)
{
    const u8* p = (const u8*)&ipv4;
    return Be32(p);
}
u32 ToNet(u32 ip)
{
    u32 out;
    PutBe32((u8*)&out, ip);
    return out;
}

bool SeqBefore(u32 a, u32 b) { return (s32)(a - b) < 0; }

u32 SumWords(const u8* data, size_t len, u32 sum)
{
    for (size_t i = 0; i + 1 < len; i += 2) sum += Be16(&data[i]);
    if (len & 1) sum += (u32)data[len - 1] << 8;
    return sum;
}

u16 FoldSum(u32 sum)
{
    while (sum >> 16) sum = (sum & 0xFFFF) + (sum >> 16);
    return (u16)~sum;
}

// --- Frames to the console ----------------------------------------------------------------

void Queue(std::vector<u8>&& frame)
{
    if (ToDS.size() >= kMaxQueuedFrames)
    {
        ToDS.pop_front();
    }
    ToDS.push_back(std::move(frame));
}

// An IPv4 packet from (src) to the console (dst), with a TCP/UDP checksum when it has one.
void SendIP(u32 src, u32 dst, u8 protocol, const u8* l4, size_t l4len, bool broadcast = false)
{
    std::vector<u8> f(14 + 20 + l4len);
    u8* eth = f.data();
    if (broadcast || !HaveClientMAC) memset(eth, 0xFF, 6);
    else memcpy(eth, ClientMAC, 6);
    memcpy(eth + 6, kRouterMAC, 6);
    PutBe16(eth + 12, 0x0800);

    u8* ip = eth + 14;
    ip[0] = 0x45;
    ip[1] = 0;
    PutBe16(ip + 2, (u16)(20 + l4len));
    PutBe16(ip + 4, IPv4ID++);
    PutBe16(ip + 6, 0x4000); // don't fragment
    ip[8] = 64;
    ip[9] = protocol;
    PutBe16(ip + 10, 0);
    PutBe32(ip + 12, src);
    PutBe32(ip + 16, dst);
    PutBe16(ip + 10, FoldSum(SumWords(ip, 20, 0)));

    u8* body = ip + 20;
    memcpy(body, l4, l4len);
    if (protocol == 6 || protocol == 17)
    {
        u32 sum = SumWords(ip + 12, 8, 0) + protocol + (u32)l4len;
        int at = protocol == 6 ? 16 : 6;
        PutBe16(body + at, 0);
        u16 check = FoldSum(SumWords(body, l4len, sum));
        if (protocol == 17 && check == 0) check = 0xFFFF;
        PutBe16(body + at, check);
    }
    Counters.FramesOut++;
    Queue(std::move(f));
}

void SendUDP(u32 src, u16 sport, u32 dst, u16 dport, const u8* data, size_t len, bool broadcast = false)
{
    std::vector<u8> udp(8 + len);
    PutBe16(&udp[0], sport);
    PutBe16(&udp[2], dport);
    PutBe16(&udp[4], (u16)(8 + len));
    PutBe16(&udp[6], 0);
    if (len) memcpy(&udp[8], data, len);
    SendIP(src, dst, 17, udp.data(), udp.size(), broadcast);
}

// --- ARP ------------------------------------------------------------------------------------

void HandleARP(const u8* frame, int len)
{
    if (len < 14 + 28) return;
    const u8* arp = frame + 14;
    if (Be16(arp) != 1 || Be16(arp + 2) != 0x0800 || arp[4] != 6 || arp[5] != 4) return;
    if (Be16(arp + 6) != 1) return; // requests only
    u32 sender = Be32(arp + 14);
    u32 target = Be32(arp + 24);
    // the console asking about itself (duplicate address check): nobody else is here
    if (target == sender || target == kClientIP) return;

    std::vector<u8> f(14 + 28);
    memcpy(&f[0], arp + 8, 6);
    memcpy(&f[6], kRouterMAC, 6);
    PutBe16(&f[12], 0x0806);
    u8* r = &f[14];
    PutBe16(r, 1);
    PutBe16(r + 2, 0x0800);
    r[4] = 6;
    r[5] = 4;
    PutBe16(r + 6, 2);
    memcpy(r + 8, kRouterMAC, 6); // every address on the far side of the router is the router
    PutBe32(r + 14, target);
    memcpy(r + 18, arp + 8, 6);
    PutBe32(r + 24, sender);
    Counters.FramesOut++;
    Queue(std::move(f));
}

// --- DHCP -----------------------------------------------------------------------------------

void HandleDHCP(const u8* bootp, size_t len)
{
    if (len < 240 || bootp[0] != 1 || Be32(bootp + 236) != 0x63825363) return;
    int type = 0;
    for (size_t i = 240; i < len;)
    {
        u8 code = bootp[i];
        if (code == 0) { i++; continue; }
        if (code == 255 || i + 1 >= len) break;
        u8 olen = bootp[i + 1];
        if (i + 2 + olen > len) break;
        if (code == 53 && olen >= 1) type = bootp[i + 2];
        i += 2 + olen;
    }
    int reply;
    if (type == 1) reply = 2;      // discover -> offer
    else if (type == 3) reply = 5; // request -> ack
    else if (type == 8) reply = 5; // inform -> ack
    else return;

    std::vector<u8> out(300, 0);
    out[0] = 2;
    out[1] = 1;
    out[2] = 6;
    memcpy(&out[4], bootp + 4, 4);   // xid
    memcpy(&out[10], bootp + 10, 2); // flags
    if (type == 8) memcpy(&out[12], bootp + 12, 4);
    else PutBe32(&out[16], kClientIP);
    PutBe32(&out[20], kRouterIP);
    memcpy(&out[28], bootp + 28, 16); // chaddr
    PutBe32(&out[236], 0x63825363);
    u8* o = &out[240];
    *o++ = 53; *o++ = 1; *o++ = (u8)reply;
    *o++ = 54; *o++ = 4; PutBe32(o, kRouterIP); o += 4;
    if (type != 8) { *o++ = 51; *o++ = 4; PutBe32(o, 86400); o += 4; }
    *o++ = 1; *o++ = 4; PutBe32(o, 0xFFFFFF00); o += 4;
    *o++ = 3; *o++ = 4; PutBe32(o, kRouterIP); o += 4;
    *o++ = 6; *o++ = 4; PutBe32(o, kDNSIP); o += 4;
    *o++ = 255;

    Counters.DhcpReplies++;
    SendUDP(kRouterIP, 67, 0xFFFFFFFF, 68, out.data(), out.size(), true);
}

// --- DNS ------------------------------------------------------------------------------------

struct DnsQuery
{
    int Handle;
    u32 Server, Client; // the server the console asked (we answer as it), and the console
    u16 ClientPort;
    std::vector<u8> Question; // the query's header and question
    u64 Start;
};
std::vector<DnsQuery> Dns;

// rcode: 0 ok, 2 server failure, 3 no such name.
void AnswerDNS(const DnsQuery& q, int rcode, u32 ipv4)
{
    std::vector<u8> r(q.Question);
    bool answer = rcode == 0 && ipv4 != 0;
    r[2] = 0x80 | (r[2] & 0x01); // response, keep RD
    r[3] = 0x80 | (u8)rcode;     // recursion available
    PutBe16(&r[4], 1);
    PutBe16(&r[6], answer ? 1 : 0);
    PutBe16(&r[8], 0);
    PutBe16(&r[10], 0);
    if (answer)
    {
        const u8 rr[16] = {0xC0, 0x0C, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 0, 0, 0, 0};
        size_t at = r.size();
        r.insert(r.end(), rr, rr + 16);
        PutBe32(&r[at + 12], ipv4);
    }
    SendUDP(q.Server, 53, q.Client, q.ClientPort, r.data(), r.size());
}

void HandleDNS(u32 src, u16 sport, u32 dst, const u8* msg, size_t len)
{
    if (len < 12 || (msg[2] & 0x80) || Be16(msg + 4) != 1) return;
    // the name, in plain labels (queries never compress)
    std::string name;
    size_t i = 12;
    for (;;)
    {
        if (i >= len) return;
        u8 l = msg[i++];
        if (l == 0) break;
        if (l > 63 || i + l > len || name.size() + l > 253) return;
        if (!name.empty()) name += '.';
        name.append((const char*)&msg[i], l);
        i += l;
    }
    if (i + 4 > len) return;
    u16 qtype = Be16(&msg[i]), qclass = Be16(&msg[i + 2]);
    i += 4;

    DnsQuery q;
    q.Handle = 0;
    q.Server = dst;
    q.Client = src;
    q.ClientPort = sport;
    q.Question.assign(msg, msg + i);
    q.Start = NowUs();
    Counters.DnsQueries++;

    if (qclass != 1 || qtype != 1)
    {
        AnswerDNS(q, 0, 0); // IPv4 only: nothing else to tell
        return;
    }
    if (!Inet || Dns.size() >= kMaxDns)
    {
        AnswerDNS(q, 2, 0);
        return;
    }
    // the console retrying a query that is still being looked up
    for (const DnsQuery& p : Dns)
        if (p.Client == src && p.ClientPort == sport && p.Question == q.Question) return;

    q.Handle = Inet->resolve(Inet->ctx, name.c_str());
    if (q.Handle <= 0)
    {
        AnswerDNS(q, 2, 0);
        return;
    }
    printf("Tamariba: the DS looks up %s\n", name.c_str());
    Dns.push_back(q);
}

void PollDNS()
{
    u64 now = NowUs();
    for (size_t i = 0; i < Dns.size();)
    {
        DnsQuery& q = Dns[i];
        u32 addr = 0;
        int res = Inet ? Inet->resolve_result(Inet->ctx, q.Handle, &addr) : TAMARIBA_INET_ERROR;
        if (res == TAMARIBA_INET_AGAIN && now - q.Start < kDnsTimeoutUs)
        {
            i++;
            continue;
        }
        if (res == TAMARIBA_INET_OK && addr != 0) AnswerDNS(q, 0, FromNet(addr));
        else if (res == TAMARIBA_INET_AGAIN) AnswerDNS(q, 2, 0);
        else AnswerDNS(q, 3, 0);
        Dns.erase(Dns.begin() + i);
    }
}

// --- UDP ------------------------------------------------------------------------------------

struct UdpFlow
{
    u16 Port; // the console's
    u32 Client;
    int Handle;
    u64 LastUse;
};
std::vector<UdpFlow> Udp;

void HandleUDP(u32 src, u32 dst, const u8* udp, size_t len)
{
    if (len < 8) return;
    u16 sport = Be16(udp), dport = Be16(udp + 2);
    size_t ulen = Be16(udp + 4);
    if (ulen < 8 || ulen > len) return;
    const u8* data = udp + 8;
    size_t dlen = ulen - 8;

    if (dport == 67) return HandleDHCP(data, dlen);
    if (dport == 53) return HandleDNS(src, sport, dst, data, dlen);
    if (!Inet || dst == 0xFFFFFFFF || (dst & 0xFFFFFF00) == kSubnet || (dst >> 28) == 0xE) return;

    UdpFlow* flow = nullptr;
    for (UdpFlow& f : Udp)
        if (f.Port == sport) flow = &f;
    if (!flow)
    {
        if (Udp.size() >= kMaxUdp)
        {
            // the one quiet the longest makes way
            size_t oldest = 0;
            for (size_t i = 1; i < Udp.size(); i++)
                if (Udp[i].LastUse < Udp[oldest].LastUse) oldest = i;
            Inet->udp_close(Inet->ctx, Udp[oldest].Handle);
            Udp.erase(Udp.begin() + oldest);
        }
        int handle = Inet->udp_open(Inet->ctx, sport);
        if (handle <= 0) return;
        Udp.push_back(UdpFlow{sport, src, handle, NowUs()});
        flow = &Udp.back();
        Counters.UdpFlows++;
    }
    flow->Client = src;
    flow->LastUse = NowUs();
    Inet->udp_sendto(Inet->ctx, flow->Handle, ToNet(dst), dport, data, dlen);
    Counters.BytesUp += dlen;
}

void PollUDP()
{
    u64 now = NowUs();
    u8 buf[1500];
    for (size_t i = 0; i < Udp.size();)
    {
        UdpFlow& f = Udp[i];
        for (int n = 0; n < 16; n++)
        {
            u32 from = 0;
            u16 port = 0;
            int got = Inet->udp_recvfrom(Inet->ctx, f.Handle, &from, &port, buf, 1500 - 28);
            if (got < 0) break;
            SendUDP(FromNet(from), port, f.Client, f.Port, buf, (size_t)got);
            Counters.BytesDown += got;
            f.LastUse = now;
        }
        if (now - f.LastUse > kIdleTimeoutUs)
        {
            Inet->udp_close(Inet->ctx, f.Handle);
            Udp.erase(Udp.begin() + i);
            continue;
        }
        i++;
    }
}

// --- TCP ------------------------------------------------------------------------------------

enum TcpFlag : u8 { FIN = 0x01, SYN = 0x02, RST = 0x04, PSH = 0x08, ACK = 0x10 };

enum TcpState
{
    Tcp_Connecting,  // the exit is connecting; the console's SYN waits for our SYN-ACK
    Tcp_SynReceived, // SYN-ACK sent
    Tcp_Established,
    Tcp_Closed       // to be removed
};

struct TcpConn
{
    u32 Client, Remote;
    u16 ClientPort, RemotePort;
    int Handle;
    TcpState State;
    u32 ISS;
    u32 SndUna, SndNxt; // toward the console; SndUna is the first byte of ToDS
    u32 RcvNxt;         // from the console
    u32 DSWindow;
    u32 DSMSS;
    std::vector<u8> ToInet, ToDS;
    bool InetClosed;  // the far side is done sending
    bool HandleOpen;
    bool FinSent;     // FIN is at SndNxt-1 when sent
    bool DSFin;
    u64 Start, LastSend, LastActivity;
    u64 RTO;
};
std::vector<TcpConn> Tcp;

void SendTCP(TcpConn& c, u8 flags, u32 seq, const u8* data, size_t len)
{
    u8 seg[20 + 4 + 1460];
    size_t hlen = (flags & SYN) ? 24 : 20;
    PutBe16(seg, c.RemotePort);
    PutBe16(seg + 2, c.ClientPort);
    PutBe32(seg + 4, seq);
    PutBe32(seg + 8, (flags & ACK) ? c.RcvNxt : 0);
    seg[12] = (u8)((hlen / 4) << 4);
    seg[13] = flags;
    size_t space = c.ToInet.size() < kBufferToInet ? kBufferToInet - c.ToInet.size() : 0;
    PutBe16(seg + 14, (u16)(space > 0xFFFF ? 0xFFFF : space));
    PutBe16(seg + 16, 0);
    PutBe16(seg + 18, 0);
    if (flags & SYN)
    {
        seg[20] = 2;
        seg[21] = 4;
        PutBe16(seg + 22, (u16)kOurMSS);
    }
    if (len) memcpy(seg + hlen, data, len);
    SendIP(c.Remote, c.Client, 6, seg, hlen + len);
}

// A reset for a segment that belongs to no connection we know.
void SendReset(u32 src, u32 dst, const u8* tcp, size_t seglen)
{
    u8 flags = tcp[13];
    if (flags & RST) return;
    u8 seg[20];
    PutBe16(seg, Be16(tcp + 2));
    PutBe16(seg + 2, Be16(tcp));
    u32 seq = Be32(tcp + 4);
    if (flags & ACK)
    {
        PutBe32(seg + 4, Be32(tcp + 8));
        PutBe32(seg + 8, 0);
        seg[13] = RST;
    }
    else
    {
        PutBe32(seg + 4, 0);
        PutBe32(seg + 8, seq + (u32)seglen + ((flags & SYN) ? 1 : 0) + ((flags & FIN) ? 1 : 0));
        seg[13] = RST | ACK;
    }
    seg[12] = 5 << 4;
    PutBe16(seg + 14, 0);
    PutBe16(seg + 16, 0);
    PutBe16(seg + 18, 0);
    SendIP(dst, src, 6, seg, 20);
}

void CloseHandle(TcpConn& c)
{
    if (c.HandleOpen && Inet) Inet->tcp_close(Inet->ctx, c.Handle);
    c.HandleOpen = false;
}

void Abort(TcpConn& c)
{
    SendTCP(c, RST | ACK, c.SndNxt, nullptr, 0);
    CloseHandle(c);
    c.State = Tcp_Closed;
}

void FlushToInet(TcpConn& c)
{
    while (!c.ToInet.empty() && c.HandleOpen)
    {
        int n = Inet->tcp_send(Inet->ctx, c.Handle, c.ToInet.data(), c.ToInet.size());
        if (n == TAMARIBA_INET_AGAIN || n == 0) return;
        if (n < 0)
        {
            Abort(c);
            return;
        }
        Counters.BytesUp += n;
        c.ToInet.erase(c.ToInet.begin(), c.ToInet.begin() + n);
    }
}

// Sends what the console's window allows, and our FIN once everything is through.
void Transmit(TcpConn& c, u64 now)
{
    if (c.State != Tcp_Established) return;
    for (;;)
    {
        if (c.FinSent) break;
        u32 inflight = c.SndNxt - c.SndUna;
        if (inflight >= c.ToDS.size()) break;
        u32 window = c.DSWindow > inflight ? c.DSWindow - inflight : 0;
        size_t chunk = c.ToDS.size() - inflight;
        if (chunk > c.DSMSS) chunk = c.DSMSS;
        if (chunk > window) chunk = window;
        if (chunk == 0) break;
        SendTCP(c, ACK | PSH, c.SndNxt, &c.ToDS[inflight], chunk);
        c.SndNxt += (u32)chunk;
        c.LastSend = now;
    }
    if (c.InetClosed && !c.FinSent && c.SndNxt - c.SndUna == c.ToDS.size())
    {
        SendTCP(c, FIN | ACK, c.SndNxt, nullptr, 0);
        c.SndNxt++;
        c.FinSent = true;
        c.LastSend = now;
    }
}

TcpConn* FindTcp(u32 client, u16 cport, u32 remote, u16 rport)
{
    for (TcpConn& c : Tcp)
        if (c.State != Tcp_Closed && c.Client == client && c.ClientPort == cport && c.Remote == remote && c.RemotePort == rport)
            return &c;
    return nullptr;
}

void HandleTCP(u32 src, u32 dst, const u8* tcp, size_t len)
{
    if (len < 20) return;
    size_t hlen = (tcp[12] >> 4) * 4;
    if (hlen < 20 || hlen > len) return;
    u16 sport = Be16(tcp), dport = Be16(tcp + 2);
    u32 seq = Be32(tcp + 4), ack = Be32(tcp + 8);
    u8 flags = tcp[13];
    u32 window = Be16(tcp + 14);
    const u8* data = tcp + hlen;
    size_t dlen = len - hlen;
    u64 now = NowUs();

    TcpConn* c = FindTcp(src, sport, dst, dport);
    if (!c)
    {
        if ((flags & (SYN | ACK | RST)) != SYN)
        {
            SendReset(src, dst, tcp, dlen);
            return;
        }
        if (!Inet || Tcp.size() >= kMaxTcp || (dst & 0xFFFFFF00) == kSubnet)
        {
            SendReset(src, dst, tcp, dlen);
            return;
        }
        int handle = Inet->tcp_connect(Inet->ctx, ToNet(dst), dport);
        if (handle <= 0)
        {
            SendReset(src, dst, tcp, dlen);
            return;
        }
        TcpConn n;
        n.Client = src;
        n.Remote = dst;
        n.ClientPort = sport;
        n.RemotePort = dport;
        n.Handle = handle;
        n.HandleOpen = true;
        n.State = Tcp_Connecting;
        n.ISS = NextISS;
        NextISS += 0x10000 + (u32)(now & 0xFFFF);
        n.SndUna = n.SndNxt = n.ISS;
        n.RcvNxt = seq + 1;
        n.DSWindow = window;
        n.DSMSS = 536;
        for (size_t i = 20; i + 1 < hlen;)
        {
            u8 kind = tcp[i];
            if (kind == 0) break;
            if (kind == 1) { i++; continue; }
            u8 olen = tcp[i + 1];
            if (olen < 2 || i + olen > hlen) break;
            if (kind == 2 && olen == 4) n.DSMSS = Be16(&tcp[i + 2]);
            i += olen;
        }
        if (n.DSMSS > kOurMSS) n.DSMSS = kOurMSS;
        if (n.DSMSS < 64) n.DSMSS = 64;
        n.InetClosed = n.FinSent = n.DSFin = false;
        n.Start = n.LastSend = n.LastActivity = now;
        n.RTO = kRetransmitUs;
        Tcp.push_back(std::move(n));
        Counters.TcpConnections++;
        return;
    }

    c->LastActivity = now;
    if (flags & RST)
    {
        CloseHandle(*c);
        c->State = Tcp_Closed;
        return;
    }
    if (flags & SYN)
    {
        // the console repeating its SYN: answer again once we can
        if (c->State == Tcp_SynReceived) SendTCP(*c, SYN | ACK, c->ISS, nullptr, 0);
        return;
    }
    if (c->State == Tcp_Connecting) return;
    if (!(flags & ACK)) return;

    c->DSWindow = window;
    // what the console acknowledges
    if (SeqBefore(c->SndUna, ack) && !SeqBefore(c->SndNxt, ack))
    {
        u32 acked = ack - c->SndUna;
        if (c->State == Tcp_SynReceived)
        {
            c->State = Tcp_Established;
            acked--; // our SYN
            c->SndUna++;
        }
        if (c->FinSent && ack == c->SndNxt) acked--; // our FIN
        if (acked > c->ToDS.size()) acked = (u32)c->ToDS.size();
        c->ToDS.erase(c->ToDS.begin(), c->ToDS.begin() + acked);
        c->SndUna = ack;
        c->RTO = kRetransmitUs;
        c->LastSend = now;
    }
    if (c->State == Tcp_SynReceived) return;

    // what the console sends: only in order (anything else gets our ACK again)
    bool needack = false;
    if (dlen > 0 || (flags & FIN))
    {
        needack = true;
        if (seq == c->RcvNxt && !c->DSFin)
        {
            size_t space = c->ToInet.size() < kBufferToInet ? kBufferToInet - c->ToInet.size() : 0;
            size_t take = dlen < space ? dlen : space;
            if (take > 0 && c->HandleOpen)
            {
                c->ToInet.insert(c->ToInet.end(), data, data + take);
                c->RcvNxt += (u32)take;
            }
            else if (take > 0)
            {
                c->RcvNxt += (u32)take; // the far side is gone: let the console finish
            }
            if ((flags & FIN) && take == dlen)
            {
                c->RcvNxt++;
                c->DSFin = true;
            }
        }
    }
    FlushToInet(*c);
    if (c->State == Tcp_Closed) return;
    if (needack) SendTCP(*c, ACK, c->SndNxt, nullptr, 0);
}

void PollTCP()
{
    u64 now = NowUs();
    u8 buf[4096];
    for (size_t i = 0; i < Tcp.size(); i++)
    {
        TcpConn& c = Tcp[i];
        if (c.State == Tcp_Closed) continue;

        if (c.State == Tcp_Connecting)
        {
            int st = Inet->tcp_state(Inet->ctx, c.Handle);
            if (st == TAMARIBA_INET_AGAIN)
            {
                if (now - c.Start > kConnectTimeoutUs) Abort(c);
                continue;
            }
            if (st != TAMARIBA_INET_OK)
            {
                Abort(c);
                continue;
            }
            c.State = Tcp_SynReceived;
            SendTCP(c, SYN | ACK, c.ISS, nullptr, 0);
            c.SndNxt = c.ISS + 1;
            c.LastSend = now;
            continue;
        }

        FlushToInet(c);
        if (c.State == Tcp_Closed) continue;

        // the far side's data, as far as our buffer goes
        while (c.HandleOpen && !c.InetClosed && c.ToDS.size() < kBufferToDS)
        {
            size_t room = kBufferToDS - c.ToDS.size();
            int n = Inet->tcp_recv(Inet->ctx, c.Handle, buf, room < sizeof(buf) ? room : sizeof(buf));
            if (n > 0)
            {
                c.ToDS.insert(c.ToDS.end(), buf, buf + n);
                Counters.BytesDown += n;
                c.LastActivity = now;
                continue;
            }
            if (n == TAMARIBA_INET_AGAIN || n == 0) break;
            c.InetClosed = true; // closed, or gone: the console gets what came, then FIN
        }

        // the console finished and everything it sent went out: the far side can close
        if (c.DSFin && c.ToInet.empty() && c.HandleOpen)
        {
            CloseHandle(c);
            c.InetClosed = true;
        }

        // nothing acknowledged for a while: send again from the first unacknowledged byte
        if (c.SndNxt != c.SndUna && now - c.LastSend > c.RTO)
        {
            if (c.State == Tcp_SynReceived)
            {
                SendTCP(c, SYN | ACK, c.ISS, nullptr, 0);
            }
            else
            {
                c.SndNxt = c.SndUna;
                c.FinSent = false;
            }
            c.LastSend = now;
            if (c.RTO < 8000000) c.RTO *= 2;
        }

        Transmit(c, now);

        bool done = c.FinSent && c.SndUna == c.SndNxt && c.DSFin;
        if (done || now - c.LastActivity > kIdleTimeoutUs)
        {
            CloseHandle(c);
            c.State = Tcp_Closed;
        }
    }
    for (size_t i = 0; i < Tcp.size();)
    {
        if (Tcp[i].State == Tcp_Closed) Tcp.erase(Tcp.begin() + i);
        else i++;
    }
}

// --- IPv4 -----------------------------------------------------------------------------------

void HandleICMP(u32 src, u32 dst, const u8* icmp, size_t len)
{
    // pings to the router and its DNS answer (the console's own checks); nothing beyond
    if (len < 8 || icmp[0] != 8 || (dst != kRouterIP && dst != kDNSIP)) return;
    std::vector<u8> r(icmp, icmp + len);
    r[0] = 0;
    PutBe16(&r[2], 0);
    PutBe16(&r[2], FoldSum(SumWords(r.data(), r.size(), 0)));
    SendIP(dst, src, 1, r.data(), r.size());
}

void HandleIPv4(const u8* frame, int len)
{
    if (len < 14 + 20) return;
    const u8* ip = frame + 14;
    if ((ip[0] >> 4) != 4) return;
    size_t ihl = (ip[0] & 0xF) * 4;
    size_t total = Be16(ip + 2);
    if (ihl < 20 || total < ihl || total > (size_t)len - 14) return;
    if (Be16(ip + 6) & 0x3FFF) return; // fragments: the console doesn't send them
    u32 src = Be32(ip + 12), dst = Be32(ip + 16);
    const u8* body = ip + ihl;
    size_t blen = total - ihl;
    switch (ip[9])
    {
    case 1: HandleICMP(src, dst, body, blen); break;
    case 6: HandleTCP(src, dst, body, blen); break;
    case 17: HandleUDP(src, dst, body, blen); break;
    }
}

void DropAll(bool resetConsole)
{
    for (TcpConn& c : Tcp)
    {
        if (resetConsole && c.State != Tcp_Connecting && c.State != Tcp_Closed)
            SendTCP(c, RST | ACK, c.SndNxt, nullptr, 0);
    }
    Tcp.clear();
    Udp.clear();
    for (const DnsQuery& q : Dns)
        if (resetConsole) AnswerDNS(q, 2, 0);
    Dns.clear();
}

}

void SetInet(const tamariba_inet_interface* inet)
{
    if (inet && inet->version != TAMARIBA_NET_VERSION) inet = nullptr;
    if (inet == Inet) return;
    // the old interface's handles are gone with it: the console's connections end
    DropAll(Inited);
    Inet = inet;
    printf("Tamariba: the DS's internet %s\n", inet ? "goes through the Plaza" : "is off");
}

bool Init()
{
    DropAll(false);
    ToDS.clear();
    HaveClientMAC = false;
    Inited = true;
    return true;
}

void DeInit()
{
    if (Inet)
    {
        for (TcpConn& c : Tcp) CloseHandle(c);
        for (UdpFlow& f : Udp) Inet->udp_close(Inet->ctx, f.Handle);
    }
    DropAll(false);
    ToDS.clear();
    Inited = false;
}

int SendPacket(u8* data, int len)
{
    if (len < 14 || len > 2048) return 0;
    Counters.FramesIn++;
    memcpy(ClientMAC, data + 6, 6);
    HaveClientMAC = true;
    u16 type = Be16(data + 12);
    if (type == 0x0806) HandleARP(data, len);
    else if (type == 0x0800) HandleIPv4(data, len);
    return len;
}

int RecvPacket(u8* data)
{
    if (Inet)
    {
        PollDNS();
        PollUDP();
        PollTCP();
    }
    if (ToDS.empty()) return 0;
    std::vector<u8> f = std::move(ToDS.front());
    ToDS.pop_front();
    memcpy(data, f.data(), f.size());
    return (int)f.size();
}

Stats GetStats()
{
    return Counters;
}

}

// The frontend's internet exit (tamariba_net.h): set after retro_load_game while the player
// has online services on, NULL when they go off.
extern "C" __attribute__((visibility("default")))
void retro_tamariba_set_inet(const struct tamariba_inet_interface* inet)
{
    TamaribaLAN::SetInet(inet);
}
