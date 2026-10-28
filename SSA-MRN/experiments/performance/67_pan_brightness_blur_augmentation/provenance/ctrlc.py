"""Send CTRL+C to the console of a given process (the 67 trainer), so its SIGINT handler saves latest.pt and exits. No forced kill."""
import ctypes, sys, json, time
pid = int(sys.argv[1]); out = sys.argv[2]
k = ctypes.windll.kernel32
res = {'pid': pid}
k.FreeConsole()
res['attach'] = bool(k.AttachConsole(pid))
if res['attach']:
    k.SetConsoleCtrlHandler(None, True)          # this helper ignores the event itself
    res['sent'] = bool(k.GenerateConsoleCtrlEvent(0, 0))   # CTRL_C_EVENT to every process on that console
    time.sleep(1); k.FreeConsole()
else:
    res['error'] = k.GetLastError()
open(out, 'w').write(json.dumps(res))
