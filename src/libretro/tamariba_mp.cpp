// Tamariba: DS local wireless through the Tamariba Plaza (tamariba_mp.h).

#include "tamariba_mp.h"
#include "tamariba_net.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

namespace TamaribaMP
{

namespace
{

// Each datagram: this header, then (a piece of) the frame as melonDS sends it.
//   0  'T' 'D'   magic
//   2  version   (1)
//   3  kind      FrameKind
//   4  round     u32 LE: the MP command this frame belongs to (commands, replies, acks)
//   8  stamp     u32 LE: the sender's emulated microseconds when it sent the frame
//  12  frag      high nibble: piece index, low nibble: pieces (1: whole frame)
//  13  fragid    u8, counts up per fragmented frame
//  14  length    u16 LE: the whole frame's length
const int kHeaderSize = 16;
const int kMaxPiece = TAMARIBA_LINK_MAX_DATAGRAM - kHeaderSize;
const int kMaxFrame = 2048;
const u8 kVersion = 1;
const size_t kMaxQueued = 256;

struct Frame
{
    int From;
    u8 Kind;
    u32 Round;
    u32 Stamp;
    std::vector<u8> Data;
};

struct Reassembly
{
    bool Busy = false;
    u8 Id = 0, Count = 0, Have = 0; // Have: bit i: piece i is here
    u16 Length = 0;
    u8 Kind = 0;
    u32 Round = 0, Stamp = 0;
    u8 Data[kMaxFrame];
};

const tamariba_link_interface* Link = nullptr;
std::deque<Frame> Queue;
Reassembly Pieces[32];
u8 NextFragId = 0;

u32 HostRound = 0;      // our last MP command, as host
int ReplySeat = -1;     // who sent the last MP command we took: our replies go to it
u32 ClientRound = 0;    // the round of that command
Stats Counters = {};

u16 Get16(const u8* p) { return (u16)(p[0] | (p[1] << 8)); }
u32 Get32(const u8* p) { return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24); }
void Put16(u8* p, u16 v) { p[0] = (u8)v; p[1] = (u8)(v >> 8); }
void Put32(u8* p, u32 v) { p[0] = (u8)v; p[1] = (u8)(v >> 8); p[2] = (u8)(v >> 16); p[3] = (u8)(v >> 24); }

u64 NowUs()
{
    return (u64)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

u8 Classify(const u8* data, int len)
{
    if (len < 12 + 2) return Frame_Other;
    u16 framectl = Get16(&data[12]);
    if ((framectl & 0x000C) != 0x0008) return Frame_Other; // not a data frame
    switch (framectl & 0xE7FF)
    {
    case 0x0228: return Frame_Cmd;
    case 0x0118:
    case 0x0158: return Frame_Reply;
    case 0x0218: return Frame_Ack;
    default: return Frame_Other;
    }
}

int OthersInRoom()
{
    if (!Link) return 0;
    u32 members = Link->members(Link->ctx) & ~(1u << Link->self(Link->ctx));
    int n = 0;
    for (; members; members &= members - 1) n++;
    return n;
}

// The longest round trip to anyone in the room, in microseconds (0: not known yet).
u32 LongestRtt()
{
    if (!Link || !Link->rtt_us) return 0;
    u32 members = Link->members(Link->ctx);
    int self = Link->self(Link->ctx);
    u32 longest = 0;
    for (int seat = 0; seat < 32; seat++)
    {
        if (seat == self || !(members & (1u << seat))) continue;
        u32 rtt = Link->rtt_us(Link->ctx, seat);
        if (rtt > longest) longest = rtt;
    }
    return longest;
}

void Enqueue(int from, u8 kind, u32 round, u32 stamp, const u8* data, int len)
{
    if (kind == Frame_Reply && round != HostRound)
    {
        Counters.Dropped++; // a reply to an earlier command that came too late
        return;
    }
    if (Queue.size() >= kMaxQueued)
    {
        Queue.pop_front();
        Counters.Dropped++;
    }
    Queue.push_back(Frame{from, kind, round, stamp, std::vector<u8>(data, data + len)});
    Counters.Received++;
}

// Takes every datagram the link has into the queue.
void Pump()
{
    if (!Link) return;
    u8 buf[TAMARIBA_LINK_MAX_DATAGRAM];
    for (;;)
    {
        int from = -1;
        int n = Link->recv(Link->ctx, &from, buf, sizeof(buf));
        if (n <= 0) return;
        if (n < kHeaderSize || buf[0] != 'T' || buf[1] != 'D' || buf[2] != kVersion || from < 0 || from >= 32)
        {
            Counters.Dropped++;
            continue;
        }
        u8 kind = buf[3];
        u32 round = Get32(&buf[4]);
        u32 stamp = Get32(&buf[8]);
        int index = buf[12] >> 4, count = buf[12] & 0xF;
        u8 fragid = buf[13];
        int length = Get16(&buf[14]);
        int piece = n - kHeaderSize;
        if (count == 0 || index >= count || length > kMaxFrame || index * kMaxPiece + piece > length)
        {
            Counters.Dropped++;
            continue;
        }
        if (count == 1)
        {
            if (piece != length) { Counters.Dropped++; continue; }
            Enqueue(from, kind, round, stamp, &buf[kHeaderSize], length);
            continue;
        }
        Reassembly& r = Pieces[from];
        if (!r.Busy || r.Id != fragid)
        {
            r.Busy = true;
            r.Id = fragid;
            r.Count = (u8)count;
            r.Have = 0;
            r.Length = (u16)length;
            r.Kind = kind;
            r.Round = round;
            r.Stamp = stamp;
        }
        memcpy(&r.Data[index * kMaxPiece], &buf[kHeaderSize], piece);
        r.Have |= (u8)(1 << index);
        if (r.Have == (u8)((1 << r.Count) - 1))
        {
            r.Busy = false;
            Enqueue(from, r.Kind, r.Round, r.Stamp, r.Data, r.Length);
        }
    }
}

int QueuedReplies()
{
    int n = 0;
    for (const Frame& f : Queue)
        if (f.Kind == Frame_Reply && f.Round == HostRound) n++;
    return n;
}

void Reset()
{
    Queue.clear();
    for (Reassembly& r : Pieces) r.Busy = false;
    ReplySeat = -1;
    ClientRound = 0;
}

}

void SetLink(const tamariba_link_interface* link)
{
    if (link && link->version != TAMARIBA_NET_VERSION) link = nullptr;
    Link = link;
    Reset();
    printf("Tamariba: local wireless %s\n", link ? "through the Plaza" : "off");
}

bool Active()
{
    return Link != nullptr && OthersInRoom() > 0;
}

int Send(const u8* data, int len, u64 timestamp)
{
    if (!Link || len <= 0 || len > kMaxFrame) return 0;

    u8 kind = Classify(data, len);
    int to = TAMARIBA_LINK_EVERYONE;
    u32 round = 0;
    switch (kind)
    {
    case Frame_Cmd:
        round = ++HostRound;
        // replies to the previous command can't count any more
        for (auto it = Queue.begin(); it != Queue.end();)
            it = it->Kind == Frame_Reply ? Queue.erase(it) : it + 1;
        break;
    case Frame_Ack:
        round = HostRound;
        break;
    case Frame_Reply:
        // only the MP host listens to replies (the hardware answers a command right after
        // taking it)
        round = ClientRound;
        if (ReplySeat >= 0) to = ReplySeat;
        break;
    }

    int count = (len + kMaxPiece - 1) / kMaxPiece;
    u8 fragid = count > 1 ? NextFragId++ : 0;
    u8 buf[TAMARIBA_LINK_MAX_DATAGRAM];
    for (int i = 0; i < count; i++)
    {
        int offset = i * kMaxPiece;
        int piece = len - offset < kMaxPiece ? len - offset : kMaxPiece;
        buf[0] = 'T';
        buf[1] = 'D';
        buf[2] = kVersion;
        buf[3] = kind;
        Put32(&buf[4], round);
        Put32(&buf[8], (u32)timestamp);
        buf[12] = (u8)((i << 4) | count);
        buf[13] = fragid;
        Put16(&buf[14], (u16)len);
        memcpy(&buf[kHeaderSize], &data[offset], piece);
        if (Link->send(Link->ctx, to, buf, kHeaderSize + piece) != 0) return 0;
    }
    Counters.Sent++;
    return len;
}

int Recv(u8* data, bool repliesOnly, RxInfo* info)
{
    Pump();
    for (auto it = Queue.begin(); it != Queue.end();)
    {
        if (repliesOnly && !(it->Kind == Frame_Reply && it->Round == HostRound))
        {
            ++it;
            continue;
        }
        if (!repliesOnly && it->Kind == Frame_Reply)
        {
            // a reply outside the host's reply window came too late for it
            it = Queue.erase(it);
            Counters.Dropped++;
            continue;
        }
        int len = (int)it->Data.size();
        memcpy(data, it->Data.data(), len);
        if (info)
        {
            info->Kind = it->Kind;
            info->Timestamp = it->Stamp;
        }
        if (it->Kind == Frame_Cmd)
        {
            ReplySeat = it->From;
            ClientRound = it->Round;
        }
        Queue.erase(it);
        return len;
    }
    return 0;
}

int WaitReplies(int count)
{
    if (!Link) return 0;
    int others = OthersInRoom();
    if (count > others) count = others;
    if (count <= 0) return 0;

    // The replies need one round trip through the Plaza, plus however long the clients take
    // to reach the command (at worst most of a frame of emulation).
    u32 rtt = LongestRtt();
    u32 timeout = rtt ? rtt + rtt / 2 + 25000 : 80000;
    if (timeout < 30000) timeout = 30000;
    if (timeout > 200000) timeout = 200000;

    u64 start = NowUs();
    Counters.ReplyWaits++;
    for (;;)
    {
        Pump();
        int have = QueuedReplies();
        u64 elapsed = NowUs() - start;
        if (have >= count || elapsed >= timeout)
        {
            if (have < count) Counters.ReplyTimeouts++;
            Counters.ReplyWaitUs += elapsed;
            return have;
        }
        if (Link->wait(Link->ctx, (u32)(timeout - elapsed)) < 0) return QueuedReplies();
    }
}

bool WaitHostFrame()
{
    if (!Link || OthersInRoom() == 0) return false;
    u32 rtt = LongestRtt();
    u32 timeout = 300000 + 2 * rtt;

    u64 start = NowUs();
    Counters.HostWaits++;
    for (;;)
    {
        Pump();
        u64 elapsed = NowUs() - start;
        if (!Queue.empty())
        {
            Counters.HostWaitUs += elapsed;
            return true;
        }
        if (elapsed >= timeout)
        {
            Counters.HostTimeouts++;
            Counters.HostWaitUs += elapsed;
            return false;
        }
        if (Link->wait(Link->ctx, (u32)(timeout - elapsed)) < 0) return false;
    }
}

Stats GetStats()
{
    return Counters;
}

}

// The frontend's room (tamariba_net.h): set after retro_load_game, NULL when the room ends.
extern "C" __attribute__((visibility("default")))
void retro_tamariba_set_link(const struct tamariba_link_interface* link)
{
    TamaribaMP::SetLink(link);
}
