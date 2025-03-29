// 模型测试页面样式
import { alpha } from '@mui/material/styles';

export const playgroundStyles = (theme) => ({
  container: {
    p: 3, 
    height: 'calc(100vh - 64px)', 
    display: 'flex', 
    flexDirection: 'column'
  },
  mainPaper: {
    p: 3, 
    flex: 1, 
    display: 'flex', 
    flexDirection: 'column', 
    mb: 2, 
    borderRadius: 2
  },
  controlsContainer: {
    mb: 2
  },
  clearButton: {
    height: '56px'
  },
  divider: {
    mb: 2
  },
  emptyStateBox: {
    flex: 1, 
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center',
    mb: 2,
    p: 2,
    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)',
    borderRadius: 1
  },
  chatContainer: {
    flex: 1, 
    mb: 2
  },
  modelPaper: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    border: `1px solid ${theme.palette.divider}`,
    borderRadius: 1,
    overflow: 'hidden'
  },
  modelHeader: {
    p: 1, 
    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.05)' : 'primary.light',
    color: theme.palette.mode === 'dark' ? 'white' : 'white',
    fontWeight: 'medium',
    textAlign: 'center',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  modelChatBox: {
    flex: 1,
    overflowY: 'auto',
    p: 2,
    bgcolor: theme.palette.mode === 'dark' ? 'rgba(255,255,255,0.03)' : 'rgba(0,0,0,0.02)'
  },
  emptyChatBox: {
    display: 'flex', 
    justifyContent: 'center', 
    alignItems: 'center', 
    height: '100%'
  },
  inputContainer: {
    display: 'flex', 
    gap: 1, 
    mt: 2
  },
  sendButton: {
    minWidth: '120px'
  }
});
