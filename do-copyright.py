#!/usr/bin/env python3

#
# (c) Copyright 2020 CORSIKA Project, corsika-project@lists.kit.edu
#
# This software is distributed under the terms of the 3-clause BSD license.
# See file LICENSE for a full version of the license.
#

"""
Script to crawl all files (hpp,inl,cpp) in a directory structure and
check if there is an initial comment block within each file that
resembles the CORSIKA 8 copyright notice.

Exceptions can be specified in `excludeDirs` and `excludedFiles`.
"""


import os
import sys, getopt
import re

"""
 Note: this is technically used as a mutliline regexp
"""
text = """/*
 * (c) Copyright YEAR CORSIKA Project, corsika-project@lists.kit.edu
 *
 * This software is distributed under the terms of the 3-clause BSD license.
 * See file LICENSE for a full version of the license.
 */\n
"""

"""
Debug settings are 0: nothing, 1: checking, 2: filesystem
"""
Debug = 0

excludeDirs = ["./modules", "./externals", "build", "install", "git", "framework/units", "Random123"]
excludeFiles = ['PhysicalConstants.h','CorsikaFenvOSX.cc', 'sgn.h', 'quartic.h']

extensions = [".cpp", ".inl", ".hpp"]

"""
justCheck: T: only checking, F: also changing files
"""
justCheck = True
"""
foundMissing: global variable set to True in case of any needed action
"""
foundMissing = False
"""
updateMessage: T: update message, put preserve the YEAR, F: no special action
"""
updateMessage = False

"""
forYear: replace this with year for copyright notice via command line
"""
forYear="YEAR"


###############################################
#
def checkNote(filename, justCheck, forYear, updateMessage):
    """
    function to check, if the file 'filename' contains an exact copy
    of the copyright notice defined above.
    The function also checks for eventual multiple (maybe conflicting)
    copyright notices in the same file.

    If 'justCheck' is True, the file will never be changed, otherwise
    the function will attempt to put the correct notice exactly once
    at the top of the file. The copyright YEAR will be replace with
    'forYear', e.g. forYear="2019" or forYear="2018-2020", etc.

    If 'updateMessage' is True, then the value of Year will be determined
    from the previous copyright message, if possible. Thus, the copyright year
    is preserved.

    The global variable 'foundMissing' is set to True in the event
    where any changes are identified, BUT not implemented.
    """
    global foundMissing

    if Debug>0:
        print ("***********************************************")
        print ("file: " + filename )

    startNote = []
    endNote = []

    """ read input file into lines """
    lines = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file.readlines():
            lines.append(line)
        file.close()

    """ 0:before comment block, #1 in comment block, #2 found copyright """
    searchStatus = 0
    blockStart = 0
    for iLine in range(len(lines)):
        line = lines[iLine]
        if "/*" in line and searchStatus==0:
            searchStatus = 1
            blockStart = iLine
        if "copyright" in line.lower() and searchStatus>0:
            searchStatus = 2
        if "*/" in line:
            if searchStatus>=2:
                startNote.append(blockStart)
                endNote.append(iLine)
            searchStatus = 0
        iLine += 1

    if Debug>0:
        txt = "states: n=" + str(len(startNote))
        for i in range(len(startNote)):
            txt += ",  [" + str(startNote[i]) + "-" + str(endNote[i]) + "]"
        print ("stats: " + txt)

    """ now check if first copyright notices is already identical... """
    isSame = False
    prevYear = ""
    if len(startNote)>0:
        isSame = True
        noteLines = text.split('\n')
        for iLine in range(len(noteLines)-2):
            if startNote[0]+iLine >= len(lines):
                isSame = False
                break

            # only if updateMessage is True:
            # read the YEAR from old message to use it for new message (->update message)
            if updateMessage:
                regexYear = re.compile(r'.*Copyright\s+(\d\d\S*)\s.*')
                matchYear = re.match(regexYear, lines[startNote[0]+iLine].strip(" \n"))
                if matchYear and len(matchYear.groups())>0:
                    prevYear = str(matchYear.groups()[0])

            # compare old with new message here:
            regex = re.compile(re.escape(noteLines[iLine].strip(" \n")).replace('YEAR','(..+)'))
            match = re.match(regex, lines[startNote[0]+iLine].strip(" \n"))
            if not match:
                if isSame:
                    print ("needs update: " + filename + "\n   [diff] new=\'" + noteLines[iLine] +
                           "\' vs old=\'" + lines[startNote[0]+iLine].rstrip('\n') + "\'")
                isSame = False
                foundMissing = True

    if Debug>0:
        print ("isSame=" + str(isSame) + " " + str(len(startNote)))

    """ check if notice is the same, or we need to remove multiple notices... """
    if isSame and len(startNote)<=1:
        return

    if (len(startNote)==0):
        print ("No copyright note in file: " + filename)

    """ either we found a file with no copyright, or with wrong copyright notice here """
    if justCheck:
        foundMissing = True
        return

    """ add (new) copyright notice here: """
    print ("   [file] " + filename + " --> backup to " + filename+".bak")
    os.rename(filename, filename+".bak")

    with open(filename, "w") as file:

        if len(prevYear)>0:
            textReplace = re.sub(r"Copyright YEAR ", "Copyright " + prevYear + " ", text)
        else:
            textReplace = re.sub(r"Copyright YEAR ", "Copyright " + forYear + " ", text)
        file.write(textReplace)

        skip = False
        for iLine in range(len(lines)):

            inBlock = False
            for iBlock in range(len(startNote)):
                if iLine>=startNote[iBlock] and iLine<=endNote[iBlock]:
                    print ("   [remove " + str(iBlock) + "] " + (lines[iLine]).strip())
                    inBlock = True
                    skip = True

            if inBlock:
                continue

            """ if line after comment is empty, also remove it """
            if lines[iLine].strip() != "":
                skip = False

            if not skip:
                file.write(lines[iLine])

        file.close()


###############################################
#
def next_file(dir_name, files, justCheck, forYear, updateMessage):
    """
    check files: loops over list of files,
    excludes if wished, process otherwise
    """
    for check in excludeDirs :
        if check in dir_name:
            if Debug>1:
                print ("exclude-dir: " + check, dir_name)
            return True
    for check in files :
        if (os.path.isdir(check)):
            continue
        filename, file_extension = os.path.splitext(check)
        if '#' in check or '~' in check:
            continue
        excludeThisFile=False
        for check2 in excludeFiles :
            if check2 in check:
                if Debug>1:
                    print ("exclude: " + check2)
                excludeThisFile=True
        if excludeThisFile:
            continue
        if file_extension in extensions:
            checkNote(os.path.join(dir_name, check), justCheck, forYear, updateMessage)
        else:
            if Debug>1:
                print ("exclude-extension: " + os.path.join(dir_name, check))


###############################################
#
def main(argv):
   """
    the main program
   """
   global justCheck, foundMissing, updateMessage, Debug, forYear
   justCheck = True
   updateMessage = False
   Debug = 0
   try:
      opts, args = getopt.getopt(argv, "cnuA:hd:", ["check", "no-check", "update-message", "add=", "help", "debug="])
   except getopt.GetoptError:
      print ('do-copyright.py [--check/--no-check] [--add=YEAR] [--update-message] [--debug=0]')
      sys.exit(2)
   for opt, arg in opts:
      if opt in ("-h", "--help"):
         print ('do-copyright.py [--check/--no-check] [--add=YEAR] [--update-message] [--debug=0]')
         sys.exit()
      elif opt in ("-c", "--check"):
         justCheck = True
      elif opt in ("-n", "--no-check"):
         justCheck = False
      elif opt in ("-u", "--update-message"):
         updateMessage = True
         print ('Preserve YEAR of existing message, where possible.')
      elif opt in ("-A", "--add"):
         justCheck = False
         forYear = str(arg)
         print ('Adding \'Copyright ' + forYear + '\' notice, where needed. ')
      elif opt in ("-d", "--debug"):
         Debug = int(arg)

   if justCheck:
       print ('Only checking. No changes. See \'do-copyright.py -h\' for options.')
   else:
       print ('** WARN: THIS WILL MODIFY YOUR FILES! ** ')

   for root, dirs, files in os.walk('./'):
       next_file(root, files, justCheck, forYear, updateMessage)


###############################################
#
if __name__ == "__main__":
   """
   main python entry point
   """
   main(sys.argv[1:])

   if justCheck and foundMissing:
       """ found any need for action """
       sys.exit(-1)
   print ("Finished")
   sys.exit(0)
