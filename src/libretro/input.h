#ifndef _INPUT_H
#define _INPUT_H

#include "types.h"

enum TouchMode
{
   Disabled,
   Mouse,
   Touch,
   Joystick,
};

struct InputState
{
   bool touching;
   int touch_x, touch_y;
   TouchMode current_touch_mode;
   bool cursor_visible = false;

   bool holding_noise_btn = false;
   bool swap_screens_btn = false;
   bool lid_closed = false;
};

extern InputState input_state;

// Tamariba: a touch screen on another device (the phone companion), set through
// melonds_set_touch() before each frame. While pressed it touches the DS screen at (x, y)
// in DS pixels, in any touch mode and any layout (Top Only too), beside the cursor.
struct ExternalTouch
{
   bool pressed = false;
   int x = 0, y = 0;
};

extern ExternalTouch external_touch;

bool cursor_enabled(InputState *state);

extern bool libretro_supports_bitmasks;

void update_input(InputState *state);

#endif
