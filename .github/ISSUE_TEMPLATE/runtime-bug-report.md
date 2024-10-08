---
name: Runtime bug report
about: Report on an issue while running PyAV.
title: "The FOO does not BAR."
assignees: ''

---

**IMPORTANT:** Be sure to replace all template sections {{ like this }} or your issue may be discarded.


## Overview

{{ A clear and concise description of what the bug is. }}


## Expected behavior

{{ A clear and concise description of what you expected to happen. }}


## Actual behavior

{{ A clear and concise description of what actually happened. }}

Traceback:
```
{{ Include complete tracebacks if there are any exceptions. }}
```


## Investigation

{{ What you did to isolate the problem. }}


## Reproduction

{{ Steps to reproduce the behavior. If the problem is media specific, include a link to it. Only send media that you have the rights to. }}


## Versions

- OS: {{ e.g. macOS 10.13.6 }}
- PyAV runtime:
```
{{ Complete output of `python -m av --version`. If this command won't run, you are likely dealing with the build issue and should use the appropriate template. }}
```
- [ ] I am/tried using the binary wheels
- [ ] I compiled from source

## Research

I have done the following:

- [ ] Checked the [PyAV documentation](https://pyav.basswood-io.com)
- [ ] Searched on [Google](https://www.google.com/search?q=pyav+how+do+I+foo)
- [ ] Searched on [Stack Overflow](https://stackoverflow.com/search?q=pyav)
- [ ] Looked through [old GitHub issues](https://github.com/PyAV-Org/PyAV/issues?&q=is%3Aissue)

## Additional context

{{ Add any other context about the problem here. }}
