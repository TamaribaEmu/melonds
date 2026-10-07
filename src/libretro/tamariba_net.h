/* Tamariba's network interfaces for emulator cores (docs/online-play-design-2026-10-06.md).
 *
 * A plain C header, so the cores (C and C++) can include it. Each fork that uses it carries an
 * exact copy (like libretro.h); this file is the original, and a change here is a change in
 * every copy (docs/fork-changes-for-tamariba-2026-10-04.md lists them).
 *
 * Two interfaces, both optional for a core. Tamariba looks for the setters softly (a core
 * without them simply has no online play of this kind):
 *
 *   void retro_tamariba_set_link(const struct tamariba_link_interface *link);
 *       Online room play between Tamariba players: datagrams to the other players in the room,
 *       through the Tamariba Plaza (never peer to peer, so no player learns another's address).
 *       Called after retro_load_game when the game joins a room, and with NULL when it leaves.
 *
 *   void retro_tamariba_set_inet(const struct tamariba_inet_interface *inet);
 *       The emulated console's own internet traffic (Pretendo, Wiimmfi, WiiLink WFC), leaving
 *       through the Plaza's internet exit. Called after retro_load_game when the player has
 *       online services on, and with NULL when they go off. Without it the core must not open
 *       host sockets for the console's traffic at all.
 *
 * Threading: every function may be called from any thread, never blocks unless it says so,
 * and stays valid until the setter is called again (with NULL or another interface). The
 * frontend calls the setters between retro_run calls, never during one.
 */
#ifndef TAMARIBA_NET_H
#define TAMARIBA_NET_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TAMARIBA_NET_VERSION 1

/* The largest datagram send() takes and recv() returns, in bytes (it fits one relay datagram
 * under a 1280-byte path MTU). Protocols with their own packet size (ENet) must stay at or
 * below it. */
#define TAMARIBA_LINK_MAX_DATAGRAM 1200
#define TAMARIBA_LINK_EVERYONE (-1)

/* --- Room play --------------------------------------------------------------------------- */

struct tamariba_link_interface {
    unsigned version; /* TAMARIBA_NET_VERSION */
    void *ctx;        /* passed back to every function */

    /* This player's seat in the room: 0 is the host, then 1, 2, 3 by joining order. */
    int (*self)(void *ctx);
    /* The seats taken right now, as bits (bit i: seat i is in the room and connected). */
    uint32_t (*members)(void *ctx);
    /* Sends one datagram to seat to (or TAMARIBA_LINK_EVERYONE). Unreliable and unordered,
     * like UDP. Returns 0, or -1 when it can't be sent now (no room, too big, queue full). */
    int (*send)(void *ctx, int to, const void *data, size_t size);
    /* Takes the next datagram that arrived: copies it to buf (at most cap bytes), sets *from
     * to the sender's seat and returns its size; 0 when nothing is waiting; -1 when the room
     * is gone. Never waits. */
    int (*recv)(void *ctx, int *from, void *buf, size_t cap);
    /* Waits until a datagram arrives or timeout_us passes (0: just checks). Returns 1 when one
     * is waiting, 0 on timeout, -1 when the room is gone. The one call that blocks. */
    int (*wait)(void *ctx, uint32_t timeout_us);
    /* The round trip to seat (through the Plaza), in microseconds; 0 when not known yet. */
    uint32_t (*rtt_us)(void *ctx, int seat);
};

/* --- The console's internet traffic -------------------------------------------------------- */

/* Handles are small positive numbers; 0 and negative values are never handles.
 * Byte order: IPv4 addresses are in network byte order, ports in host byte order.
 * resolve_result frees its handle once it returns anything but TAMARIBA_INET_AGAIN; tcp_send
 * may take fewer bytes than given (send the rest later); a UDP handle is usable as soon as
 * udp_open returns it. There is no listen/accept: the console only connects out. */
enum tamariba_inet_status {
    TAMARIBA_INET_OK = 0,
    TAMARIBA_INET_AGAIN = -1,   /* not yet: try again later (non-blocking) */
    TAMARIBA_INET_CLOSED = -2,  /* the other side closed, or the exit went away */
    TAMARIBA_INET_REFUSED = -3, /* refused (by the destination, or by the Plaza's rules) */
    TAMARIBA_INET_ERROR = -4    /* anything else */
};

struct tamariba_inet_interface {
    unsigned version; /* TAMARIBA_NET_VERSION */
    void *ctx;

    /* The console's public address as the internet sees it (the exit's), IPv4 in network
     * byte order, for games that ask; 0 when not known yet. */
    uint32_t (*public_ipv4)(void *ctx);

    /* Name lookup, answered by the Plaza (the player's chosen services decide where Nintendo's
     * old host names go). Starts a lookup and returns a handle; poll resolve_result until it
     * stops returning TAMARIBA_INET_AGAIN. On success *ipv4 is set (network byte order). */
    int (*resolve)(void *ctx, const char *host);
    int (*resolve_result)(void *ctx, int handle, uint32_t *ipv4);

    /* TCP. connect returns a handle at once; tcp_state says TAMARIBA_INET_AGAIN while it
     * connects, then OK or an error. send/recv return bytes moved, or a status (AGAIN when
     * nothing can move now). close frees the handle. */
    int (*tcp_connect)(void *ctx, uint32_t ipv4, uint16_t port);
    int (*tcp_state)(void *ctx, int handle);
    int (*tcp_send)(void *ctx, int handle, const void *data, size_t size);
    int (*tcp_recv)(void *ctx, int handle, void *buf, size_t cap);
    void (*tcp_close)(void *ctx, int handle);

    /* UDP. udp_open gives a handle with its own public port at the exit (local_port is the
     * port the console asked for, kept when it can be). sendto/recvfrom move one datagram;
     * recvfrom returns its size, or AGAIN when none is waiting. */
    int (*udp_open)(void *ctx, uint16_t local_port);
    int (*udp_sendto)(void *ctx, int handle, uint32_t ipv4, uint16_t port, const void *data, size_t size);
    int (*udp_recvfrom)(void *ctx, int handle, uint32_t *ipv4, uint16_t *port, void *buf, size_t cap);
    void (*udp_close)(void *ctx, int handle);
};

#ifdef __cplusplus
}
#endif

#endif /* TAMARIBA_NET_H */
