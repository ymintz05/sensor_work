import time, math, argparse
import pandas as pd
from psychopy import visual, core
from psychopy.hardware import keyboard
from ds8r_state_set import get_state, set_demand

# modifiable in arduino only 
STIM_DURATION = 30.0

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
    text="Pay attention to the following stimulus",
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
    rating_value.text = ""

    while True:
        start_msg.draw()
        win.flip()
        keys = kb.getKeys(waitRelease=False)
        if keys:
            return

def rate_stim():
    kb.clearEvents()
    rating_value.text = ""

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
        
def mol_increasing(init_current=5.0, current_interval=0.2, current_cap=8.0):

    current_rating = []

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
            return current_rating

        set_demand(current_prepared)

        # stim prompt
        stim_msg.draw()
        win.flip()
        core.wait(1)

        # check current match with ds8r & arduino-triggered fire protocol
        if not math.isclose(float(get_state("demand")), current_prepared, abs_tol = 0.05):
            raise RuntimeError("requested current does not match stim current")
        #fire(current_prepared)
        core.wait(STIM_DURATION)

        current_administered = current_prepared

        # rate stim prompt, record vals
        rating_temp = rate_stim()
        current_rating.append([current_administered, rating_temp])
        rating = rating_temp

        temp += 1

    return current_rating

def mol_decreasing(init_current=8.0, current_interval=0.2):

    current_rating = []

    rating = float('inf')
    temp = 0 
    current_administered = -1
    current_prepared = -1

    while rating > 5:

        if temp == 0: 
            current_prepared = init_current
        else:
            current_prepared -= current_interval

        set_demand(current_prepared)

        # stim prompt
        stim_msg.draw()
        win.flip()
        core.wait(1)

        # check current match with ds8r & arduino-triggered fire protocol
        if not math.isclose(float(get_state("demand")), current_prepared, abs_tol = 0.05):
            raise RuntimeError("requested current does not match stim current")
        #fire(current_prepared)
        core.wait(STIM_DURATION)

        current_administered = current_prepared

        # rate stim prompt, record vals
        rating_temp = rate_stim()
        current_rating.append([current_administered, rating_temp])
        rating = rating_temp

        temp += 1

    return current_rating

    
def main(
        participant_initials, 

        mol_inc_vns_init_curr = 5.0,
        mol_inc_vns_curr_ivl = 0.2, 
        mol_inc_vns_current_cap = 8.0,

        mol_inc_sham_init_current = 5.0,
        mol_inc_sham_curr_ivl = 0.2, 
        mol_inc_sham_curr_cap = 8.0,

        mol_dec_vns_init_current = 5.0,
        mol_dec_vns_curr_ivl = 0.2, 

        mol_dec_sham_init_current = 5.0,
        mol_dec_sham_curr_ivl = 0.2, 
        ):
    

    wait_any_key()

    # vns protocol
    current_inc_vns = mol_increasing(
                                    mol_inc_vns_init_curr, 
                                    mol_inc_vns_curr_ivl, 
                                    mol_inc_vns_current_cap)
    wait_any_key()

    current_dec_vns = mol_decreasing(mol_dec_vns_init_current, 
                                    mol_dec_vns_curr_ivl)
    wait_any_key()
    
    df_inc_vns = pd.DataFrame(current_inc_vns, columns=["current", "rating"])
    df_dec_vns = pd.DataFrame(current_dec_vns, columns=["current", "rating"])
    df_inc_vns["direction"] = "increasing"
    df_dec_vns["direction"] = "decreasing"
    df_inc_vns["condition"] = "vns"
    df_dec_vns["condition"] = "vns"

    # sham protocol
    current_inc_sham = mol_increasing(mol_inc_sham_init_current, 
                                    mol_inc_sham_curr_ivl, 
                                    mol_inc_sham_curr_cap)
    wait_any_key()

    current_dec_sham = mol_decreasing(mol_dec_sham_init_current, 
                                    mol_dec_sham_curr_ivl)
    wait_any_key()

    df_inc_sham = pd.DataFrame(current_inc_sham, columns=["current", "rating"])
    df_dec_sham = pd.DataFrame(current_dec_sham, columns=["current", "rating"])
    df_inc_sham["direction"] = "increasing"
    df_dec_sham["direction"] = "decreasing" 
    df_inc_sham["condition"] = "sham"
    df_dec_sham["condition"] = "sham"

    df_all = pd.concat(
        [df_inc_vns, df_dec_vns, df_inc_sham, df_dec_sham],
        ignore_index=True
    )

    df_all.to_excel(f"{participant_initials}_current_settings.xlsx", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initials", "-i", type=str, required=True)
    args = parser.parse_args()

    main(args.initials)

# saves file per participant. experimental fxn would take in file and preload vals. 
#col 1 current intensity, col 2, can append to a numpy within the fxn, def 
# start at the safe cap, decrease, stop at value 5 to take, avg current for final threshold 
# 30s firing 
# stimulus page should say "pay attn to stim, also remove prev rating"

#decreasing: highest cap, NOT THEIRs 
    
    
