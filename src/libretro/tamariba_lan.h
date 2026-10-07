// Tamariba: the internet behind melonDS's access point (WifiAP.cpp's "melonAP"), through the
// Tamariba Plaza's internet exit.
//
// The access point hands the console's Ethernet frames to Platform::LAN_* (platform.cpp),
// which land here: a small user-mode router in the spirit of melonDS's LAN_Socket, but with
// the Plaza in place of host sockets (tamariba_net.h's tamariba_inet_interface, set by the
// frontend with retro_tamariba_set_inet). It answers ARP and DHCP itself (the console gets
// 10.64.0.16, router 10.64.0.1, DNS 10.64.0.2: the "automatic" settings of a Nintendo WFC
// connection), answers every DNS query (to any server, so a console with Wiimmfi's DNS typed
// in works too) through the Plaza, and carries TCP and UDP to the exit. Without the inet
// interface no host socket is ever opened: names don't resolve and connections are refused,
// so a game says it could not connect.

#ifndef TAMARIBA_LAN_H
#define TAMARIBA_LAN_H

#include "types.h"

struct tamariba_inet_interface;

namespace TamaribaLAN
{

void SetInet(const tamariba_inet_interface* inet);

bool Init();
void DeInit();

// An Ethernet frame from the console. Returns len.
int SendPacket(u8* data, int len);
// The next Ethernet frame for the console (at most 2048 bytes); 0 when none.
int RecvPacket(u8* data);

// For tests and the log.
struct Stats
{
    u64 FramesIn, FramesOut;
    u64 DnsQueries, DhcpReplies, TcpConnections, UdpFlows;
    u64 BytesUp, BytesDown;
};
Stats GetStats();

}

#endif
