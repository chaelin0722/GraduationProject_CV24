package com.example.server;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;

public class MainActivity extends AppCompatActivity {
    String rcvIp, rcvPort, rcvPacket;
    String ind;
    Button btn;

    private final Handler mHandler = new Handler();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn = (Button) findViewById(R.id.btn);
        btn.setOnClickListener(V -> {
            Toast.makeText(this, "server set",Toast.LENGTH_SHORT).show();

            //UDP 수신 쓰레드 설정 및 시작부분
            ReceiveData rcvServer = new ReceiveData(9090);
            rcvServer.start();
/*
            @Override
            public void onClick(View v){  OpenPopUp(ind);  }
 */
        });
    }
    class ReceiveData extends Thread {
        DatagramSocket rSocket;

        //UDP 수신 포트를 열기(원하는 port 아무거나)
        public ReceiveData(int port){
            try {
                //수신 소켓 생성
                rSocket = new DatagramSocket(port);
            } catch (SocketException e) {
                e.printStackTrace();
            }
        }

        public void run(){
            try {
                while (true){
                    //받을 패킷 생성
                    byte[] buf = new byte[1024];

                    //패킷으로 변경 (바이트 버퍼, 버퍼길이)
                    DatagramPacket packet = new DatagramPacket(buf, buf.length);

                    //데이터 수신 대기
                    rSocket.receive(packet);

                    //클라이언트에서 보낸 메시지 저장!
                    ind = new String(packet.getData(),0,packet.getLength());
                    OpenPopUp(ind);

                    //패킷을 보낸 상대방의 ip를 저장함
                    InetAddress ina = packet.getAddress();
                    rcvIp = ina.toString();

                    //패킷을 보낸 상대방의 port를 저장함
                    int inp = packet.getPort();
                    rcvPort = String.valueOf(inp);

                    //수신받은 데이터를 문자열로 변환
                    rcvPacket = new String(buf);

                    //에코하기 위해 송신용 패킷을 만듦 (받은 패킷, 패킷 길이, 상대방 IP, 상대방 PORT)
                    packet = new DatagramPacket(buf, buf.length, ina, inp);

                    //이 기기로 UDP송신을 한 상대방에게 받은패킷을 돌려줌
                    rSocket.send(packet);

                    //데이터를 받았다면 UI로 표현해주기 위해 runnable 사용
                    //handler.post(rnb);

                    //쓰레드를 인터럽트로 종료시키기 위해 sleep을 사용함
                    sleep(20);
                }
            }catch (InterruptedException e){
                rcvPacket = e.toString();
                //handler.post(rnb);
            }
            catch (Exception e){
                rcvPacket = e.toString();
                //handler.post(rnb);
            }
        }
    }


        private void OpenPopUp(CharSequence message){
            Intent intent = new Intent(MainActivity.this, FullInfoPop.class);
            intent.putExtra("message", message);
            startActivity(intent);
        }

}