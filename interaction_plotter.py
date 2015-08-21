# -*- coding: utf-8 -*- 
#    PyInteraph, a software suite to analyze interactions and interaction network in structural ensembles.
#    Copyright (C) 2013 Matteo Tiberti <matteo.tiberti@gmail.com>, Gaetano Invernizzi, Yuval Inbar, 
#    Matteo Lambrughi, Gideon Schreiber,  Elena Papaleo <elena.papaleo@unimib.it> <elena.papaleo@bio.ku.dk>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.

#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from Tkinter import *
from pymol import cmd, stored, CmdException

from numpy import array
from pymol.cgo import CYLINDER
import tkMessageBox
import Pmw
import tkFileDialog
import tkColorChooser
import os
import ConfigParser as cp

def __init__(self):
    self.menuBar.addmenuitem('Plugin', 'command',
                                       'Interaction plotter',
                                       label = 'Interaction plotter',
                                       command = lambda s=self : InteractionPlotterTk(s))
    
class InteractionPlotterTk:

    def open_input_file(self):
        self.fname= tkFileDialog.askopenfilename()
        self.fname_input.setentry(self.fname)

    def open_cg_file(self):
        self.cg_fname= tkFileDialog.askopenfilename()
        self.cg_fname_input.setentry(self.cg_fname)

    def quick_file_validator(self, s):
        if s == '':
            return Pmw.PARTIAL
        elif os.path.isfile(s):
            return Pmw.OK
        elif os.path.exists(s):
            return Pmw.PARTIAL
        else:
            return Pmw.PARTIAL

    def posfloat_validator(self, s):
        try:
            n = float(s)
        except:
            return Pmw.PARTIAL
        if n >= 0.0:
            return Pmw.OK
        return Pmw.PARTIAL
        
    def object_validator(self, s):
        if s in cmd.get_names("all")+["all"]:
            return Pmw.OK
        return Pmw.PARTIAL
        
    def not_object_validator(self, s):
        if s not in cmd.get_names("all")+["all"]:
            return Pmw.OK
        return Pmw.PARTIAL

    def interpret_color(self, c):
        if cmd.get_version()[1] < 1:
            return cmd._interpret_color(str(c))
        return cmd._interpret_color(cmd, str(c))
        
    def update_color_frames(self):
        rgbSmin = "#%s" % "".join(["%02x"%(x*255.) for x in self.colorRGB_min])
        rgbSmax = "#%s" % "".join(["%02x"%(x*255.) for x in self.colorRGB_max])
        self.color_min_input.configure(background=rgbSmin)
        self.color_max_input.configure(background=rgbSmax)

    def tk_color_dialog(self, col):
        rgb = tuple([x*255 for x in col])
        color = tkColorChooser.Chooser(
            initialcolor=rgb,title='Choose color').show()
        if color[0] is not None:
            return [color[0][0]/255.,
                    color[0][1]/255.,
                    color[0][2]/255.]
        else:
            return col

    def tk_min_color_dialog(self):
        self.colorRGB_min = self.tk_color_dialog(self.colorRGB_min)
        self.update_color_frames()

    def tk_max_color_dialog(self):
        self.colorRGB_max = self.tk_color_dialog(self.colorRGB_max)
        self.update_color_frames()        
        
    def make_gradient(self, name, init, end, steps):
        i0 = 0
        # look for free space !!!
        #while is_color("%s%d"%(name,i0)): i0+=1
        iS=range(i0,i0+steps)
        cS=[]
        # start creating colors
        for i in range(steps):
            g=[]
            for j in range(len(init)):
                g.append(init[j]+float(end[j]-init[j])/steps*i)
            c = "%s%d"%(name,iS[i])
            cS.append(c)
            cmd.set_color(c, g)
        return cS

    def main_runner(self, button):
        if button == 'Plot':        
            self.do_plot()
        if button == 'Close':
            self.dialog.withdraw()
        if button == 'Info':
            pass
    def show_error(self, message):
        tkMessageBox.showerror("Error", message)

    def show_warning(self, message):
        tkMessageBox.showerror("Warning", message)

    def do_plot(self):
        
        if not self.obj_input.valid():
            self.show_error("The specified input object does not exist or is not valid.")
            return
            
        if not self.out_obj_input.valid():
            self.show_error("The specified output object already exists or is not valid.")
            return            

        if not self.fname_input.valid(): 
            self.show_error("The specified input file does not exist.")
            return
        self.fname = self.fname_input.getvalue()
        try: 
            fh = open(self.fname)
        except:
            self.show_error("Could not open the input file.")
            return
        fh.close()
            
        if not self.min_input.valid():
            self.show_error("The minimum value must be a positive number.")
            return
        if not self.max_input.valid():
            self.show_error("The maximum value must be a positive number.")
            return
        self.min = float(self.min_input.getvalue())
        self.max = float(self.max_input.getvalue())
        if not self.min < self.max:
            self.show_error("Error: the persistence minimum value must be lower than the maximum.")
            return                        

        if not self.min_radius_input.valid():
            self.show_error("The minimum radius value must be a positive number.")
            return
        if not self.max_radius_input.valid():
            self.show_error("The maximum radius value must be a positive number.")
            return
        self.min_r = float(self.min_radius_input.getvalue())
        self.max_r = float(self.max_radius_input.getvalue())
        if not self.min_r < self.max_r:
            self.show_error("The radius minimum value must be lower than the maximum.")
            return                        
        
        self.obj = self.obj_input.getvalue()
        self.cg_fname = self.cg_fname_input.getvalue()
        self.fname = self.fname_input.getvalue()
        self.out_obj = self.out_obj_input.getvalue()
        

        self.KEY_ATOMS = self.parse_cg_file(self.cg_fname)
        
        self.coords, self.freqs, dowarn = self.parse_file(self.fname, obj = self.obj)

        if not self.coords:
            return 

        self.plot(self.coords, 
                  self.freqs, 
                  self.min, 
                  self.max, 
                  self.min_r, 
                  self.max_r, 
                  self.colorRGB_min,
                  self.colorRGB_max,
                  self.out_obj)

        if dowarn: 
            self.show_warning("Could not plot one ore more interactions.")

    def parse_cg_file(self, fname):

        grps_str = 'CHARGED_GROUPS'
        default_grps_str = 'default_charged_groups'

        cfg = cp.ConfigParser()

        if not cfg.read(fname):
            self.show_warning("Charge groups file not readeable or not in the right format. A standard set of charged groups will be used.")

            KEY_ATOMS = { 'sidechain':'CA',
                          'ctern':'C',
                          'cterxn':'C',
                          'ctertn':'C',
                          'nterp':'N',
                          'sc.asp.coon':'CG',
                          'sc.arg.ncnnp':'CD',
                          'sc.glu.coon':'CD',
                          'sc.his.edp':'CG',
                          'sc.lys.nzp':'CE' }
            return KEY_ATOMS

        KEY_ATOMS = {'sidechain':'CA'}
                               
        charged_groups = cfg.options(grps_str)
        charged_groups.remove(default_grps_str)
        charged_groups = [ i.strip() for i in charged_groups ]

        for i in charged_groups:
            KEY_ATOMS[i] = cfg.get(grps_str, i).split(",")[0].strip().lower()

        return KEY_ATOMS

    def parse_file(self, fname, obj='all'):
        freqs = []
        stored.pos = []        
        
        fh = open(fname, 'r')

        dowarn = False

        print self.KEY_ATOMS

        for line in fh:
            tmp, freq = line.strip().split()
            freqs.append(float(freq))
            res1_str, res2_str = tmp.split(":")
            res1 = res1_str.split("-")
            res2 = res2_str.split("-")
        
            if res1[0] == 'SYSTEM':
                res1_c = None
            else:
                res1_c = res1[0]
            if res2[0] == 'SYSTEM':
                res2_c = None
            else:
                res2_c = res2[0]
            
            if (res1_c == None and res2_c != None) or (res1_c != None and res2_c == None):
                log.error("Something's wrong with the chain names. Exiting...")
                return None
        
            res1_a = res1[1].split("_")[1].lower()
            res2_a = res2[1].split("_")[1].lower()
            
            if res1_a in self.KEY_ATOMS.keys():
                res1_a = self.KEY_ATOMS[res1_a.lower()]
            if res2_a in self.KEY_ATOMS.keys():
                res2_a = self.KEY_ATOMS[res2_a.lower()]

            print res1_a
            print res2_a
        
            res1 = res1[1].split("_")[0]
            res2 = res2[1].split("_")[0]
            
            res1_i = int(res1[:-3])
            res2_i = int(res2[:-3])

            if res1_c == None:
                res1_c_str = ""
                res2_c_str = ""
            else:
                res1_c_str = "chain %s and " % res1_c
                res2_c_str = "chain %s and " % res2_c   
            
            sel1 = "%s %s and name %s and (resi %d)" % (res1_c_str, obj, res1_a, res1_i)
            sel2 = "%s %s and name %s and (resi %d)" % (res2_c_str, obj, res2_a, res2_i)
            
            print "sel1", sel1
            print "sel2", sel2

            pos_s = len(stored.pos)
            try:
                cmd.iterate_state(1, sel1, 'stored.pos.append((x,y,z))')
                pos_1 = len(stored.pos)
                cmd.iterate_state(1, sel2, 'stored.pos.append((x,y,z))')
                pos_2 = len(stored.pos)
            except:
                self.show_error("The selected object is not valid.")
                return (None, None, None)
                
            delta_pos_1 = pos_1 - pos_s
            delta_pos_2 = pos_2 - pos_1

            this_warn = False

            if delta_pos_2 != 1 or delta_pos_1 != 1:
                for i in range(delta_pos_2):
                    stored.pos.pop()
                for i in range(delta_pos_1):
                    stored.pos.pop()
                dowarn = True
                this_warn = True

            if this_warn:
                freqs.pop()
            
        return (stored.pos, freqs, dowarn)

    
    def plot(self, coords, freqs, min, max, min_r, max_r, col_min, col_max, out_obj):
        
        widths = []
        cgo = []
        nsteps = 200
        gradient = []
        cols = []

        #gradient = self.make_gradient("sbcol", array(cmd.get_color_tuple(self.interpret_color(col_min))), array(cmd.get_color_tuple(self.interpret_color(col_max))), nsteps )
        gradient = self.make_gradient("sbcol", array(col_min), array(col_max), nsteps )
        
        for i,freq in enumerate(freqs):
            if freq < min:
                freqs[i] = min
            elif freq > max: 
                freqs[i] = max
            widths.append( (freqs[i] - min)*((max_r - min_r) / (max - min)) + min_r )
            cols.append(gradient[int(round(freqs[i]/max*(nsteps-1)))])

        for i in range(0, len(coords), 2):
            cgo.append(CYLINDER)
            cgo.extend(coords[i])
            cgo.extend(coords[i+1])
            cgo.append(widths[i/2])
            cgo.extend(cmd.get_color_tuple(self.interpret_color(cols[i/2])))
            cgo.extend(cmd.get_color_tuple(self.interpret_color(cols[i/2])))

        
            
        cmd.load_cgo(cgo,out_obj)

    def __init__(self,app):

        self.intro_text = "the Interaction Plotter plugin is part of the PyInteraph package.\nBrought to you by:\n M. Tiberti, G. Invernizzi, Y. Inbar, M. Lambrughi, G. Schreiber, E. Papaleo"


        self.fname = ""
        self.obj = "all"
        self.out_obj = "interactions"
        self.max = 100.0
        self.min = 0.0
        self.max_r = 0.2
        self.min_r = 0.002
        self.color_max = 'red'
        self.color_min = 'white'
        self.colorRGB_min = cmd.get_color_tuple(cmd._interpret_color(cmd, str(self.color_min)))
        self.colorRGB_max = cmd.get_color_tuple(cmd._interpret_color(cmd, str(self.color_max)))


        self.INSTALL_DIR = os.getenv('PYINTERAPH')

        self.root = app.root

        self.dialog = Pmw.Dialog(self.root,                                 
                                 buttons = ("Close", "Plot"),
                                 title = 'Interaction Calculator',
                                 command = self.main_runner)

        self.dialog.geometry('473x400')
        self.dialog.show()
        

        self.frame = self.dialog.interior()
        #self.frame.grid(row = 0, column = 0)

        self.top_label = Label(self.frame,
                  text = self.intro_text,
                  background = 'black',
                  foreground = 'orange',
                  width = 57, )

        self.cg_fname_input = Pmw.EntryField(self.frame,
                                    labelpos='w',
                                    label_text = 'Charged groups file',
                                    validate = {'validator': self.quick_file_validator,},
                                    value = self.fname,)

        self.fname_input = Pmw.EntryField(self.frame,
                                    labelpos='w',
                                    label_text = 'Input file',
                                    validate = {'validator': self.quick_file_validator,},
                                    value = self.fname,)
                                    
        self.obj_input = Pmw.EntryField(self.frame,
                                    label_text = 'Reference object',
                                    labelpos ='w',
                                    validate = {'validator': self.object_validator,},
                                    value = self.obj,)
                                    
        self.out_obj_input = Pmw.EntryField(self.frame,
                                    label_text = 'Plot object',
                                    labelpos ='w',
                                    validate = {'validator': self.not_object_validator,},
                                    value = self.out_obj,)
                                    
        self.cg_file_button = Button(self.frame,
                                  text = "Charged groups file...", 
                                  command=self.open_cg_file,)

        self.file_button = Button(self.frame,
                                  text = "Input file...", 
                                  command=self.open_input_file,)
        
        self.max_input = Pmw.EntryField(self.frame,
                                        labelpos='w',
                                        validate = {'validator': self.posfloat_validator,},
                                        label_text = 'Max:',
                                        value = self.max,)

        self.min_input = Pmw.EntryField(self.frame,
                                        labelpos='w',
                                        validate = {'validator': self.posfloat_validator,},
                                        label_text = 'Min:',
                                        value = self.min,)        

        self.min_radius_input = Pmw.EntryField(self.frame,
                                        labelpos='w',
                                        validate = {'validator': self.posfloat_validator,},
                                        label_text = 'Min radius:',
                                        value = self.min_r,)

        self.max_radius_input = Pmw.EntryField(self.frame,
                                        labelpos='w',
                                        validate = {'validator': self.posfloat_validator,},
                                        label_text = 'Max radius:',
                                        value = self.max_r,)        

        self.color_min_input = Button(self.frame, 
                                relief=SUNKEN, 
                                bd=2, 
                                height=1, 
                                width=1, 
                                text="Min",
                                command=self.tk_min_color_dialog)
                                
        self.color_max_input = Button(self.frame, 
                                relief=SUNKEN, 
                                bd=2, 
                                height=1, 
                                width=1, 
                                text="Max",
                                command=self.tk_max_color_dialog)
                                
        self.update_color_frames()

        if self.INSTALL_DIR == None:
            self.show_warning("The PYINTERAPH system variable is not defined.")
            self.cg_fname = ""
        else:
            self.cg_fname = self.INSTALL_DIR+"/charged_groups.ini"
            self.cg_fname_input.setentry(self.cg_fname)

        self.top_label.grid(row = 0, column = 0, columnspan = 4,  pady = 5, sticky = N, padx = 5,)

        self.cg_fname_input.grid(row = 1, column = 0, columnspan = 3, ipadx = 5, ipady = 5, sticky=E)
        self.cg_file_button.grid(row = 1, column = 3, ipadx = 5, ipady = 5, sticky=N, pady=5)

        self.fname_input.grid(row = 2, column = 0, columnspan = 3, ipadx = 5, ipady = 5, sticky=E)
        self.file_button.grid(row = 2, column = 3, ipadx = 5, ipady = 5, sticky=N, pady=5)
        
        self.obj_input.grid(row = 3, column = 0, columnspan = 4, ipadx = 5, ipady = 5, sticky=N)
        self.out_obj_input.grid(row = 9, column = 0, columnspan = 4, ipadx = 5, ipady = 5, sticky=N)

        self.min_input.grid(row = 5, column = 0, columnspan = 1, ipadx = 5, sticky = E, pady = 3, )
        self.max_input.grid(row = 6, column = 0, columnspan = 1, ipadx = 5, sticky = E, pady = 3, )
        self.min_radius_input.grid(row = 7, column = 0, columnspan = 1, sticky = E, ipady = 3, ipadx = 5)
        self.max_radius_input.grid(row = 8, column = 0, columnspan = 1, sticky = E, ipady = 3, ipadx = 5)
        self.color_min_input.grid(row = 5, column = 2, rowspan = 2, columnspan = 2, sticky = N+W+S+E, ipadx = 5, padx=10 )
        self.color_max_input.grid(row = 7, column = 2, rowspan = 2, columnspan = 2, sticky = N+W+S+E, ipadx = 5, padx=10)
    


