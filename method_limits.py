from psychopy import visual, core
from psychopy.hardware import keyboard
from ds8r_state_set import get_state, set_demand
import time

win = visual.Window(
    size=(800, 600),
    color="grey",
    units="pix"
)

kb = keyboard.Keyboard()

start_msg = visual.TextStim(
    win=win,
    text="Press any key to continue",
    color="white",
    pos=(0, 0),
    height=40
)

stim_msg = visual.TextStim(
    win=win,
    text="Stimulus",
    color="white",
    pos=(0, 0),
    height=40
)

rating_prompt = visual.TextStim(
    win=win,
    text="Rate stimulus intensity from 1 (least intense) to 9 (most intense)\n\nPress a number key 1-9",
    color="white",
    pos=(0, 40),
    height=28,
    wrapWidth=700
)

rating_value = visual.TextStim(
    win=win,
    text="",
    color="white",
    pos=(0, -60),
    height=42
)

def wait_any_key():
    kb.clearEvents()
    while True:
        start_msg.draw()
        win.flip()
        keys = kb.getKeys(waitRelease=False)
        if keys:
            return

def rate_stim():
    kb.clearEvents()

    while True:
        rating_prompt.draw()
        rating_value.draw()
        win.flip()

        keys = kb.getKeys(
            keyList=["1", "2", "3", "4", "5", "6", "7", "8", "9", "escape"],
            waitRelease=False
        )

        if keys:
            key_name = keys[0].name

            if key_name == "escape":
                win.close()
                core.quit()

            rating_value.text = f"{key_name}"
            rating_prompt.draw()
            rating_value.draw()
            win.flip()
            core.wait(0.4)

            return int(key_name)
        
def mol_increasing(init_current, current_interval, current_cap):

    rating = -1
    temp = 0 
    current_administered = -1
    current_prepared = -1

    while rating < 9:

        if temp == 0: 
            current_prepared = init_current
        else:
            current_prepared += current_interval
        
        if current_prepared > current_cap:
            return current_administered

        set_demand(current_prepared)

        stim_msg.draw()
        win.flip()
        core.wait(1)
        fire(current_prepared)
        current_administered = current_prepared
        rating = rate_stim()

        temp += 1

    return current_administered
    
def main():

    wait_any_key()
    current_vns = mol_increasing(5.0,0.2,8.0)
    current_sham = mol_increasing(5.0,0.2,8.0)
    

    
    


        

 




        










# 5mA start, continuous or phasic (40s delivery)
# rate intensity 0-10, keyboard input
# 0.2 increase per trial 
# absolute, 8mA
# whatever their max was last time, start at that 
# psychopy 