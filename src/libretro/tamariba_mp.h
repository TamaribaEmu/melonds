// Tamariba: DS local wireless between Tamariba players, through the Tamariba Plaza.
//
// melonDS sends and receives the DS's 802.11 frames through Platform::MP_* (platform.cpp);
// here they travel as datagrams to the other players of the room (tamariba_net.h's
// tamariba_link_interface, set by the frontend with retro_tamariba_set_link). Without a room
// nothing is sent and nothing arrives, and no host socket is ever opened.
//
// Local multiplayer is latency-sensitive: an MP host sends a command frame and expects every
// client's reply microseconds later. As in melonDS 0.9.5+, the host blocks (in real time)
// until the replies arrive or a timeout passes, and a client never runs further ahead of the
// host's last frame than the host itself did between its frames (Wifi.cpp's client sync), so
// each MP round costs one round trip through the Plaza. See the fork-changes doc for numbers.

#ifndef TAMARIBA_MP_H
#define TAMARIBA_MP_H

#include "types.h"

struct tamariba_link_interface;

namespace TamaribaMP
{

// What a frame is in the MP protocol (from its 802.11 frame control field).
enum FrameKind : u8
{
    Frame_Other = 0, // beacons, management, plain data
    Frame_Cmd = 1,   // MP host command (0228)
    Frame_Reply = 2, // MP client reply (0118, empty: 0158)
    Frame_Ack = 3    // MP host acknowledgement (0218)
};

// One received frame and when the sender sent it (its emulated microseconds, low 32 bits).
struct RxInfo
{
    u8 Kind;
    u32 Timestamp;
};

void SetLink(const tamariba_link_interface* link);
bool Active(); // a room is set and someone else is in it

// The frame melonDS transmits: a 12-byte TX header and the 802.11 frame. timestamp is the
// console's emulated microseconds. Returns len, or 0 when it went nowhere.
int Send(const u8* data, int len, u64 timestamp);

// The next frame waiting (copied to data, at most 2048 bytes); 0 when none. With
// repliesOnly, only an MP reply to our last command (the host's reply window).
int Recv(u8* data, bool repliesOnly, RxInfo* info);

// MP host: after sending a command, wait until `count` clients' replies are here, or until
// the timeout (derived from the room's round trips) passes. Returns how many arrived.
int WaitReplies(int count);

// MP client: wait for the next frame from the MP host. False when none came in time (the
// host stopped, paused, or left): the caller then stops syncing until the next command.
bool WaitHostFrame();

// Statistics for tests and the log.
struct Stats
{
    u64 Sent, Received, Dropped;
    u64 ReplyWaits, ReplyTimeouts, ReplyWaitUs;
    u64 HostWaits, HostTimeouts, HostWaitUs;
};
Stats GetStats();

}

#endif
