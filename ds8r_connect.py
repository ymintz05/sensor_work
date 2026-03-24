import win32com.client
import tkinter as tk
from tkinter import ttk

global DS8R

def CreateDS8RReference():
    global DS8R
    DS8R = win32com.client.Dispatch("DS8R.DS8RController")
    
def ChangeDemand(NewDemand):

    global DS8R
    Collection = DS8R.getState

    if (Collection.Count > 0):
        State = collection.Items(0)
        State.Demand = NewDemand
        DS8R.SetState(state)

def Button1_Click():
    ChangeDemand(10000)
def Button2_Click():
    ChangeDemand(25000)
def main():
 
    global DS8R
    CreateDS8RReference()

    window = tk.Tk()
    window.geometry("350x350")
    window.title("Digitimer DS8R Python Example")

    Button1 = ttk.Button(window, text="Set 10mA", command=Button1_Click)
    Button1.pack()

    Button2 = ttk.Button(window, text="Set 25mA", command=Button2_Click)
    Button2.pack()

    window.mainloop()

    DS8R = None