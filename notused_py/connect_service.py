import socketio

sio = socketio.Client()
received_data = None

@sio.on('hpo_initialized')
def on_hpo_initialized(data):
    print('HPO Initialized:', data)

@sio.on('calculation_result')
def on_calculation_result(data):
    print('Calculation Result:', data)
    received_data = data
@sio.on('disconnected')
def on_disconnected(data):
    print('Disconnected:', data)
    sio.disconnect()

def main():
    # 连接到WebSocket服务器
    sio.connect('http://192.168.3.37:5000')  # 替换'server_ip'为服务器的实际IP

    # 初始化HPO
    sio.emit('initialize_hpo', {'task_id': 167149, 'Seed': 0})

    # 发送参数进行计算
    sio.emit('calculate', {'params': [[-0.6,0.2,0.3,-0.4,-0.5,0.99,0.7,0.8,0.9,0.2], [0.1,0.2,-0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.8], [0.3,0.2,-0.4,-0.9,0.5,0.7,-0.7,0.8,0.9,0.1]]})

    # 当你想断开连接时
    sio.emit('disconnect_request')

    # 等待一会儿，直到收到所有响应或断开连接
    sio.wait()

if __name__ == '__main__':
    main()