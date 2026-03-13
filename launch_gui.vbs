Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
mainScript = scriptDir & "\main.py"
venvPythonw = scriptDir & "\.venv\Scripts\pythonw.exe"
venvPython = scriptDir & "\.venv\Scripts\python.exe"

dataHome = shell.Environment("PROCESS")("DIALOG_TXT_HOME")
If Len(dataHome) > 0 Then
    If Not fso.FolderExists(dataHome) Then
        On Error Resume Next
        fso.CreateFolder dataHome
        On Error GoTo 0
    End If
End If

shell.CurrentDirectory = scriptDir

If fso.FileExists(venvPythonw) Then
    command = """" & venvPythonw & """ """ & mainScript & """"
ElseIf fso.FileExists(venvPython) Then
    command = """" & venvPython & """ """ & mainScript & """"
Else
    command = "pythonw.exe """ & mainScript & """"
End If

shell.Run command, 0, False
